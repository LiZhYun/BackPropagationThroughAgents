"""Train the shared feedforward MARL methods through optional simulator adapters."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import math
from pathlib import Path
import platform
import random
import time
import warnings

import numpy as np
import torch

from .checkpoint import save_checkpoint
from .core import mlp
from .runner import collect_rollout, update_rollout
from .value_normalization import DEFAULT_VARIANCE_FLOOR, ValueNormalizedCritic, validate_value_loss_options


ENVIRONMENT_SEED_SCHEME = "numpy_seed_sequence_bta_sim_v1"


def training_environment_seeds(seed: int, num_envs: int) -> list[int]:
    """Derive reproducible simulator streams from the run seed and env index.

    The fixed BTA/SIM/version namespace separates these streams from the run's
    model RNG. Spawning preserves earlier environment seeds when num_envs grows.
    """
    sequence = np.random.SeedSequence([0x425441, 0x53494D, 1, seed])
    return [int(child.generate_state(1, dtype=np.uint32)[0])
            for child in sequence.spawn(num_envs)]


@dataclass
class Config:
    environment: str = "smac"
    env_config: dict = field(default_factory=dict)
    algorithm: str = "agentmixer"
    seed: int = 0
    steps: int = 100_000
    num_envs: int = 1
    rollout_steps: int = 128
    hidden_dim: int = 64
    initial_std: float = 1.0
    action_embed_dim: int = 0
    agentmixer_correlation: str = 'factorized'
    joint_distribution_float64: bool = False
    modifier_enabled: bool = True
    learning_rate: float = 5e-4
    modifier_learning_rate: float = 5e-5
    critic_learning_rate: float = 5e-4
    epochs: int = 4
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    normalize_value_targets: bool = False
    value_clip_coef: float | None = None
    value_huber_delta: float | None = None
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    normalize_observations: bool = False
    observation_clip: float = 10.0
    temperature: float = 1.0
    continuous_peer_noise: str = "fresh"
    continuous_modifier_scale: str = "residual"
    eval_seed: int = 1_000_000
    eval_episodes: int = 32
    eval_max_steps: int = 10_000
    eval_every_steps: int = 0
    eval_window_steps: int = 0
    eval_windows: int = 0
    checkpoint_every_updates: int = 10
    device: str = "cpu"
    threads: int = 1

    def validate(self):
        if self.environment not in ("smac", "mujoco", "prey", "football"):
            raise ValueError("environment must be smac, mujoco, prey, or football")
        if self.algorithm not in ("mappo", "arppo", "bppo", "agentmixer"):
            raise ValueError("unknown algorithm")
        if not isinstance(self.env_config, dict) or "seed" in self.env_config:
            raise ValueError("env_config must be a dict without seed; use the top-level seed fields")
        for name in ("steps", "num_envs", "rollout_steps", "hidden_dim", "epochs", "eval_episodes", "eval_max_steps", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("eval_every_steps", "checkpoint_every_updates", "eval_window_steps", "eval_windows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if bool(self.eval_window_steps) != bool(self.eval_windows):
            raise ValueError('fixed-window evaluation requires both eval_window_steps and eval_windows')
        if self.eval_windows and self.environment != 'prey':
            raise ValueError('fixed-window evaluation currently requires Predator-Prey')
        for name in ("seed", "eval_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32)")
        environment_seeds = training_environment_seeds(self.seed, self.num_envs)
        if len(set(environment_seeds)) != self.num_envs:
            raise ValueError("derived training environment seeds collide; choose another run seed")
        if self.eval_seed in environment_seeds:
            raise ValueError("evaluation seed must differ from all training environment seeds")
        if self.steps % self.num_envs:
            raise ValueError("steps must be divisible by num_envs for an exact synchronous budget")
        for name in ("learning_rate", "modifier_learning_rate", "critic_learning_rate", "max_grad_norm", "temperature", "observation_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("entropy_coef", "value_coef"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.clip_coef < 1 or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid PPO clipping, discount, or GAE coefficient")
        for name in ("normalize_advantages", "normalize_observations", "normalize_value_targets", "modifier_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        validate_value_loss_options(self.value_clip_coef, self.value_huber_delta)
        if (isinstance(self.initial_std, bool) or not isinstance(self.initial_std, (int, float))
                or not math.isfinite(self.initial_std)
                or not math.exp(-5) <= self.initial_std <= math.exp(2)):
            raise ValueError("initial_std must be finite and within [exp(-5), exp(2)]")
        if self.environment != "mujoco" and self.initial_std != 1.0:
            raise ValueError("nondefault initial_std requires a continuous environment")
        if self.continuous_peer_noise not in ("fresh", "behavior"):
            raise ValueError("continuous_peer_noise must be fresh or behavior")
        if self.continuous_peer_noise != "fresh" and (self.environment != "mujoco" or self.algorithm != "bppo"):
            raise ValueError("behavior peer noise requires continuous BPPO")
        if self.continuous_modifier_scale not in ("residual", "bounded"):
            raise ValueError("continuous_modifier_scale must be residual or bounded")
        if self.continuous_modifier_scale != "residual" and (
            self.environment != "mujoco" or self.algorithm != "agentmixer" or not self.modifier_enabled
        ):
            raise ValueError("bounded modifier scale requires continuous AgentMixer with modifier enabled")
        if isinstance(self.action_embed_dim, bool) or not isinstance(self.action_embed_dim, int) or self.action_embed_dim < 0:
            raise ValueError('action_embed_dim must be a nonnegative integer')
        if self.action_embed_dim and (self.environment == 'mujoco' or self.algorithm not in ('arppo', 'bppo')):
            raise ValueError('action projection requires discrete AR-PPO or BPPO')
        if self.agentmixer_correlation not in ('factorized', 'joint'):
            raise ValueError('agentmixer_correlation must be factorized or joint')
        if self.agentmixer_correlation == 'joint' and (self.algorithm != 'agentmixer' or self.environment != 'prey'):
            raise ValueError('simulator joint AgentMixer currently requires Predator-Prey')
        if not isinstance(self.joint_distribution_float64, bool):
            raise ValueError('joint_distribution_float64 must be boolean')
        if self.joint_distribution_float64 and (self.algorithm != 'agentmixer' or self.agentmixer_correlation != 'joint'):
            raise ValueError('float64 joint distributions require joint AgentMixer')
        if self.algorithm != "agentmixer" and not self.modifier_enabled:
            raise ValueError("modifier_enabled=False requires algorithm=agentmixer")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")


def make_env(config: Config, seed: int):
    """Lazy imports keep all optional simulator dependencies outside the core."""
    arguments = deepcopy(config.env_config)
    arguments["seed"] = seed
    if config.environment == "smac":
        from .smac_env import SmacEnv
        return SmacEnv(arguments)
    if config.environment == "prey":
        from .prey_env import PreyEnv
        return PreyEnv(arguments)
    if config.environment == "football":
        from .football_env import FootballEnv
        return FootballEnv(arguments)
    from .mujoco_env import MujocoEnv
    return MujocoEnv(arguments)


def environment_spec(env) -> dict:
    return {
        "is_discrete": bool(env.is_discrete), "num_agents": int(env.num_agents),
        "obs_dim": int(env.obs_dim), "state_dim": int(env.state_dim),
        "action_dim": int(env.num_actions if env.is_discrete else env.action_dim),
    }


def make_policy(config: Config, env):
    if env.is_discrete and config.initial_std != 1.0:
        raise ValueError("nondefault initial_std requires a continuous environment")
    if env.is_discrete and config.continuous_modifier_scale != "residual":
        raise ValueError("bounded modifier scale requires a continuous environment")
    dimensions = (env.obs_dim, env.num_agents, env.num_actions if env.is_discrete else env.action_dim)
    scale_arguments = {} if env.is_discrete else {"initial_std": config.initial_std}
    if config.algorithm == "agentmixer":
        if env.is_discrete:
            if config.agentmixer_correlation == 'joint':
                from .agentmixer_joint import JointAgentMixerPolicy as Policy
                scale_arguments['distribution_float64'] = config.joint_distribution_float64
            else:
                from .agentmixer import AgentMixerPolicy as Policy
        else:
            from .agentmixer_continuous import ContinuousAgentMixerPolicy as Policy
            scale_arguments['modifier_scale'] = config.continuous_modifier_scale
        policy = Policy(env.obs_dim, env.state_dim, env.num_agents, dimensions[-1], config.hidden_dim,
                        **scale_arguments)
        if not config.modifier_enabled:
            policy.modifier.requires_grad_(False)
        return policy
    if config.algorithm in ("arppo", "bppo"):
        if env.is_discrete:
            from .bppo import AutoregressivePolicy as Policy
            scale_arguments['action_embed_dim'] = config.action_embed_dim
        else:
            from .bppo_continuous import ContinuousAutoregressivePolicy as Policy
    elif env.is_discrete:
        from .core import IndependentPolicy as Policy
    else:
        from .core import ContinuousIndependentPolicy as Policy
    return Policy(*dimensions, hidden_dim=config.hidden_dim, **scale_arguments)


def make_critic(config: Config, env):
    critic = mlp(env.state_dim, 1, config.hidden_dim)
    return ValueNormalizedCritic(critic) if config.normalize_value_targets else critic


def make_optimizers(config: Config, policy, critic) -> dict:
    optimizers = {"critic": torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate, eps=1e-5)}
    if config.algorithm == "bppo":
        optimizers.update({f"actor_{i}": torch.optim.Adam(actor.parameters(), lr=config.learning_rate, eps=1e-5)
                           for i, actor in enumerate(policy.actors)})
    else:
        parameters = policy.parameters()
        if config.algorithm == "agentmixer":
            parameters = [
                {"params": policy.local_actors.parameters(), "lr": config.learning_rate},
            ]
            if config.modifier_enabled:
                parameters.append({"params": policy.modifier.parameters(), "lr": config.modifier_learning_rate})
        optimizers["policy"] = torch.optim.Adam(parameters, lr=config.learning_rate, eps=1e-5)
    return optimizers


@contextmanager
def evaluation_rng(seed: int):
    """Evaluation gets its own repeatable streams and leaves training RNG intact."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@dataclass
class EpisodeStatistics:
    count: int = 0
    return_mean: float = 0.0
    return_m2: float = 0.0
    steps: int = 0
    terminated: int = 0
    truncated: int = 0
    wins: int = 0
    win_records: int = 0
    captures: int = 0
    capture_records: int = 0

    def add(self, records):
        for record in records:
            self.count += 1
            delta = record["return"] - self.return_mean
            self.return_mean += delta / self.count
            self.return_m2 += delta * (record["return"] - self.return_mean)
            self.steps += record["length"]
            self.terminated += int(record["terminated"])
            self.truncated += int(record["truncated"])
            if "battle_won" in record["info"]:
                self.win_records += 1
                self.wins += int(bool(record["info"]["battle_won"]))
            elif "match_won" in record["info"]:
                self.win_records += 1
                self.wins += int(bool(record["info"]["match_won"]))
            if "capture_success" in record["info"]:
                self.capture_records += 1
                self.captures += int(bool(record["info"]["capture_success"]))

    def summary(self):
        return {
            "count": self.count, "return_mean": self.return_mean if self.count else None,
            "return_std": math.sqrt(max(0.0, self.return_m2 / (self.count - 1))) if self.count > 1 else 0.0,
            "length_mean": self.steps / self.count if self.count else None,
            "terminated": self.terminated, "truncated": self.truncated,
            "win_rate": self.wins / self.win_records if self.win_records else None,
            "win_records": self.win_records,
            "capture_rate": self.captures / self.capture_records if self.capture_records else None,
            "capture_records": self.capture_records,
            "reward_per_environment_step": self.return_mean * self.count / self.steps if self.steps else None,
            "reward_per_step_scope": "completed_episodes_only",
        }


@torch.no_grad()
def _evaluation_action(policy, config, env, observation):
    parameter = next(policy.parameters())
    obs = torch.as_tensor(observation['obs'], device=parameter.device, dtype=parameter.dtype).unsqueeze(0)
    available = None
    if env.is_discrete:
        available = torch.as_tensor(
            observation.get('available_actions', np.ones((env.num_agents, env.num_actions), dtype=bool)),
            device=parameter.device, dtype=torch.bool).unsqueeze(0)
    if config.algorithm == 'agentmixer':
        action = policy.local_actions(obs, available) if env.is_discrete else policy.local_actions(obs)
    elif env.is_discrete:
        action, _ = policy.sample(obs, available_actions=available, deterministic=True)
    else:
        action, _ = policy.sample(obs, deterministic=True)
    return action[0].cpu().numpy().copy()


@torch.no_grad()
def evaluate_fixed_windows(policy, config: Config, expected_spec=None, normalizer=None):
    """Mean step reward over fixed windows, including unfinished final episodes.

    Each window starts in a separately seeded fresh environment and resets after
    capture or timeout. This matches the released Prey logger's reduction while
    repairing its ineffective environment seeding. Episode statistics remain a
    separate metric; window variation is not variation across training seeds.
    """
    if config.environment != 'prey' or config.eval_windows < 1 or config.eval_window_steps < 1:
        raise ValueError('positive fixed-window counts require Predator-Prey')
    if config.normalize_observations != (normalizer is not None):
        raise ValueError('fixed-window normalization must match training')
    seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in
             np.random.SeedSequence([0x425441, 0x57494E, 1, config.eval_seed]).spawn(config.eval_windows)]
    if len(set(seeds)) != len(seeds) or set(seeds).intersection(training_environment_seeds(config.seed, config.num_envs)):
        raise ValueError('fixed-window seed streams collide; choose a different evaluation seed')
    was_training = policy.training
    policy.eval()
    records = []
    try:
        with evaluation_rng(config.eval_seed):
            for index, seed in enumerate(seeds):
                env = make_env(config, seed)
                try:
                    if normalizer is not None:
                        from .normalization import NormalizedEnv
                        env = NormalizedEnv(env, normalizer.frozen_copy(), training=False)
                    if expected_spec is not None and environment_spec(env) != expected_spec:
                        raise ValueError('fixed-window environment differs from training')
                    observation, _ = env.reset()
                    total, partial_return, partial_steps = 0.0, 0.0, 0
                    completed, captures, timeouts = 0, 0, 0
                    for step in range(config.eval_window_steps):
                        action = _evaluation_action(policy, config, env, observation)
                        observation, reward, terminated, truncated, info = env.step(action)
                        if not np.isscalar(reward) or not np.isfinite(reward):
                            raise ValueError('fixed-window reward must be a finite scalar')
                        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
                            raise ValueError('fixed-window episode endings must be boolean')
                        total += float(reward)
                        partial_return += float(reward)
                        partial_steps += 1
                        if terminated or truncated:
                            completed += 1
                            captures += int(bool(info.get('capture_success', False)))
                            timeouts += int(truncated)
                            partial_return, partial_steps = 0.0, 0
                            if step + 1 < config.eval_window_steps:
                                observation, _ = env.reset()
                    records.append({'window': index, 'seed': seed, 'reward_sum': total,
                                    'steps': config.eval_window_steps, 'completed_episodes': completed,
                                    'captures': captures, 'timeouts': timeouts,
                                    'unfinished_episode_steps': partial_steps,
                                    'unfinished_episode_return': partial_return})
                finally:
                    env.close()
    finally:
        policy.train(was_training)
    steps = config.eval_windows * config.eval_window_steps
    reward = sum(record['reward_sum'] for record in records)
    return {'reward_per_environment_step': reward / steps, 'reward_sum': reward,
            'environment_steps': steps, 'steps_per_window': config.eval_window_steps,
            'window_count': config.eval_windows, 'windows': records,
            'scope': 'all_steps_in_fixed_windows_including_unfinished_episodes',
            'seed_scheme': 'numpy_seed_sequence_bta_win_v1', 'seed': config.eval_seed,
            'deterministic': True}


@torch.no_grad()
def evaluate(policy, config: Config, expected_spec: dict | None = None, normalizer=None) -> dict:
    """Evaluate fixed episodes from a fresh, separately seeded environment.

    AgentMixer executes its local actors. Autoregressive methods use ordered
    greedy/mean actions; these are reported as autoregressive execution.
    """
    if config.normalize_observations != (normalizer is not None):
        raise ValueError("normalized evaluation requires the training normalizer; raw evaluation must omit it")
    was_training = policy.training
    policy.eval()
    try:
        with evaluation_rng(config.eval_seed):
            env = make_env(config, config.eval_seed)
            try:
                if normalizer is not None:
                    from .normalization import NormalizedEnv
                    env = NormalizedEnv(env, normalizer.frozen_copy(), training=False)
                if expected_spec is not None and environment_spec(env) != expected_spec:
                    raise ValueError("evaluation environment differs from the training specification")
                records = []
                statistics = EpisodeStatistics()
                for episode in range(config.eval_episodes):
                    observation, _ = env.reset()
                    episode_return = 0.0
                    for length in range(1, config.eval_max_steps + 1):
                        action = _evaluation_action(policy, config, env, observation)
                        observation, reward, terminated, truncated, info = env.step(action)
                        if not np.isscalar(reward) or not np.isfinite(reward):
                            raise ValueError("evaluation reward must be a finite scalar")
                        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
                            raise ValueError("evaluation episode endings must be boolean")
                        episode_return += float(reward)
                        if terminated or truncated:
                            record = {"episode": episode, "return": episode_return, "length": length,
                                      "terminated": bool(terminated), "truncated": bool(truncated), "info": deepcopy(info)}
                            records.append(record)
                            statistics.add([record])
                            break
                    else:
                        raise RuntimeError("evaluation exceeded eval_max_steps without an episode ending")
                result = {**statistics.summary(), "episodes": records, "seed": config.eval_seed,
                        "environment_steps": statistics.steps,
                        "execution": "decentralized" if config.algorithm == "agentmixer" else
                                     ("autoregressive" if config.algorithm in ("bppo", "arppo") else "independent"),
                        "deterministic": True}
                if config.eval_windows:
                    result['fixed_window'] = evaluate_fixed_windows(policy, config, expected_spec, normalizer)
                    result['environment_steps'] += result['fixed_window']['environment_steps']
                return result
            finally:
                env.close()
    finally:
        policy.train(was_training)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _json(value, indent=None):
    return json.dumps(value, indent=indent, default=_json_default, allow_nan=False)


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json(value, indent=2) + "\n")
    temporary.replace(path)


def _versions():
    versions = {"python": platform.python_version()}
    for name in ("torch", "numpy", "smacv2", "pysc2", "mujoco-py", "gym", "gymnasium", "gfootball"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def critic_diagnostics(batch) -> dict:
    """Describe raw GAE targets and rollout-time predictions before updates."""
    targets = batch.returns.detach().double()
    predictions = batch.values.detach().flatten().double()
    target_variance = targets.var(unbiased=False)
    explained_variance = None
    if target_variance > 0:
        explained_variance = float(1 - (targets - predictions).var(unbiased=False) / target_variance)
    return {
        "return_mean": float(targets.mean()), "return_std": float(target_variance.sqrt()),
        "prediction_mean": float(predictions.mean()),
        "prediction_std": float(predictions.std(unbiased=False)),
        "explained_variance": explained_variance,
    }


def train(config: Config, output: Path) -> dict:
    config.validate()
    environment_seeds = training_environment_seeds(config.seed, config.num_envs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    started = time.monotonic()
    steps, update, evaluation_steps = 0, 0, 0
    envs = []
    normalizer = None
    metadata = {
        "config": asdict(config), "versions": _versions(),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(Path(__file__).parent.glob("*.py"))},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "feedforward simulator experiment; benchmark reproduction requires matched budgets and multiple seeds",
        "training_environment_seed_scheme": ENVIRONMENT_SEED_SCHEME,
        "training_environment_seeds": environment_seeds,
        "evaluation_environment_seed": config.eval_seed,
        "method_variant": {"agentmixer": "mode_consistent_modifier", "bppo": "corrected_first_order_peer_surrogate"}.get(config.algorithm, "reference"),
        "observation_normalization": {
            "enabled": config.normalize_observations,
            "convention": "running_per_agent_observation_feature_and_global_state_feature",
            "clip": config.observation_clip, "training_statistics_only": True,
            "evaluation_statistics": "frozen_training_snapshot",
            "reward_normalization": False, "value_target_normalization": config.normalize_value_targets,
        },
        "value_regression": {
            "normalize_targets": config.normalize_value_targets,
            "normalizer_estimator": "cumulative_float64_population_moments",
            "variance_floor": DEFAULT_VARIANCE_FLOOR, "statistics_scope": "shared_team_critic",
            "statistics_update": "once_per_training_rollout", "popart": False,
            "clip_coef": config.value_clip_coef, "huber_delta": config.value_huber_delta,
            "loss_units": "standardized" if config.normalize_value_targets else "raw",
            "raw_values_for_gae": True,
        },
        "checkpoint_resume": "model/optimizer/RNG saved; simulator state is not saved and exact simulator resume is unsupported",
    }
    if config.algorithm == "agentmixer" and not config.modifier_enabled:
        metadata["method_variant"] = "mode_consistent_no_modifier_control"
    if config.continuous_modifier_scale == "bounded":
        metadata["method_variant"] = "mode_consistent_bounded_modifier"
    if config.agentmixer_correlation == 'joint':
        metadata['method_variant'] = ('experimental_joint_gap_scaling' if config.modifier_enabled
                                      else 'experimental_joint_identity_control')
    _write_json(output / "config.json", metadata)
    try:
        for environment_seed in environment_seeds:
            envs.append(make_env(config, environment_seed))
        spec = environment_spec(envs[0])
        if any(environment_spec(env) != spec for env in envs[1:]):
            raise ValueError("training environments have different specifications")
        if config.normalize_observations:
            from .normalization import NormalizedEnv, ObservationNormalizer
            normalizer = ObservationNormalizer(spec["num_agents"], spec["obs_dim"], spec["state_dim"],
                                               clip=config.observation_clip)
            envs = [NormalizedEnv(env, normalizer, training=True) for env in envs]
        metadata["environment_spec"] = spec
        metadata["resolved_training_env_configs"] = [deepcopy(getattr(env, "config", {})) for env in envs]
        metadata["environment_versions"] = deepcopy(getattr(envs[0], "versions", {}))
        _write_json(output / "config.json", metadata)
        policy = make_policy(config, envs[0]).to(config.device)
        critic = make_critic(config, envs[0]).to(config.device)
        optimizers = make_optimizers(config, policy, critic)
        observations = None
        episode_statistics = EpisodeStatistics()
        next_evaluation = config.eval_every_steps if config.eval_every_steps else math.inf
        with (output / "metrics.jsonl").open("w") as metrics, (output / "episodes.jsonl").open("w") as episodes:
            def log(record):
                metrics.write(_json({"steps": steps, "update": update, "elapsed_seconds": time.monotonic() - started, **record}) + "\n")
                metrics.flush()

            def checkpoint():
                extra_state = {"observation_normalizer": normalizer.state_dict()} if normalizer is not None else None
                save_checkpoint(output / "checkpoint.pt", policy=policy, critic=critic, optimizers=optimizers,
                                config=asdict(config), step=steps, extra_state=extra_state)

            initial_evaluation = evaluate(policy, config, spec, normalizer)
            evaluation_steps += initial_evaluation["environment_steps"]
            log({"event": "evaluation", "stage": "initial", "evaluation": initial_evaluation})
            while steps < config.steps:
                rollout_steps = min(config.rollout_steps, (config.steps - steps) // config.num_envs)
                batch, observations, completed = collect_rollout(
                    envs, policy, critic, config.algorithm, rollout_steps, observations,
                    gamma=config.gamma, gae_lambda=config.gae_lambda,
                )
                # Environment interactions count even if the following update fails.
                steps += rollout_steps * config.num_envs
                update += 1
                diagnostics = critic_diagnostics(batch)
                episode_statistics.add(completed)
                for episode in completed:
                    episodes.write(_json({"steps": steps, "update": update, **episode}) + "\n")
                episodes.flush()
                losses = update_rollout(
                    policy, critic, optimizers, batch, config.algorithm, spec["is_discrete"],
                    epochs=config.epochs, clip_coef=config.clip_coef, entropy_coef=config.entropy_coef,
                    value_coef=config.value_coef, max_grad_norm=config.max_grad_norm,
                    value_clip_coef=config.value_clip_coef, value_huber_delta=config.value_huber_delta,
                    normalize_advantages=config.normalize_advantages, temperature=config.temperature,
                    continuous_peer_noise=config.continuous_peer_noise,
                )
                rollout_statistics = EpisodeStatistics()
                rollout_statistics.add(completed)
                log({"event": "update", "rollout_shape": batch.rollout_shape,
                     "rollout_return_mean": float(batch.rewards.mean()), "episodes": rollout_statistics.summary(),
                     "losses": losses, "critic_diagnostics": diagnostics})
                if config.checkpoint_every_updates and update % config.checkpoint_every_updates == 0:
                    checkpoint()
                if steps >= next_evaluation and steps < config.steps:
                    result = evaluate(policy, config, spec, normalizer)
                    evaluation_steps += result["environment_steps"]
                    log({"event": "evaluation", "stage": "periodic", "evaluation": result})
                    next_evaluation = (steps // config.eval_every_steps + 1) * config.eval_every_steps
            # Save a valid final model before the potentially expensive evaluator.
            checkpoint()
            final_evaluation = evaluate(policy, config, spec, normalizer)
            evaluation_steps += final_evaluation["environment_steps"]
            log({"event": "evaluation", "stage": "final", "evaluation": final_evaluation})
        summary = {
            "status": "completed", "steps": steps, "updates": update,
            "train_environment_steps": steps, "evaluation_environment_steps": evaluation_steps,
            "elapsed_seconds": time.monotonic() - started, "training_episodes": episode_statistics.summary(),
            "unfinished_training_episodes": sum(length > 0 for length in observations.episode_lengths),
            "initial_evaluation": initial_evaluation, "final_evaluation": final_evaluation,
        }
        _write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        _write_json(output / "summary.json", {"status": "failed", "steps": steps, "updates": update,
                    "evaluation_environment_steps": evaluation_steps, "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        for env in reversed(envs):
            try:
                env.close()
            except Exception as exc:
                warnings.warn(f"Environment close failed: {exc}", RuntimeWarning)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON containing Config fields")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are rejected")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--algorithm", choices=("mappo", "arppo", "bppo", "agentmixer"))
    args = parser.parse_args()
    values = json.loads(args.config.read_text())
    for name in ("steps", "seed", "algorithm"):
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    summary = train(Config(**values), args.output)
    print(_json(summary), flush=True)


if __name__ == "__main__":
    main()
