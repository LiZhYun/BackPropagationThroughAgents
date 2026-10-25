"""One readable training entry point for exact matrix-game experiments.

This runner validates algorithm plumbing; it is not the complete simulator
benchmark reproduction. Each joint matrix action is a terminating episode.
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import time
import uuid
import numpy as np
import torch
from torch import nn
from .core import Batch, IndependentPolicy, mlp, ppo_objective, normalize_advantages
from .envs import MatrixGame
from .checkpoint import save_checkpoint


@dataclass
class Config:
    algorithm: str = "bppo"
    game: str = "climbing"
    seed: int = 0
    steps: int = 100_000
    batch_size: int = 1024
    hidden_dim: int = 64
    learning_rate: float = 5e-4
    modifier_learning_rate: float = 5e-5
    critic_learning_rate: float = 5e-4
    epochs: int = 4
    clip_coef: float = .2
    entropy_coef: float = .01
    entropy_final: float | None = None
    entropy_decay_start: float = .5
    action_embed_dim: int = 0
    modifier_log_scale_bound: float | None = None
    agentmixer_correlation: str = "factorized"
    max_grad_norm: float = .5
    temperature: float = 1.
    eval_every: int = 10
    threads: int = 1

    def validate(self):
        if self.algorithm not in ("mappo", "arppo", "bppo", "agentmixer"):
            raise ValueError("Unknown algorithm")
        MatrixGame(self.game)
        for name in ("steps", "batch_size", "hidden_dim", "epochs", "eval_every", "threads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "modifier_learning_rate", "critic_learning_rate", "max_grad_norm", "temperature"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.clip_coef < 1 or self.entropy_coef < 0:
            raise ValueError("Invalid clipping or entropy coefficient")
        if self.entropy_final is not None and self.entropy_final < 0:
            raise ValueError("entropy_final must be nonnegative")
        if not 0 <= self.entropy_decay_start < 1:
            raise ValueError("entropy_decay_start must be in [0, 1)")
        if not isinstance(self.action_embed_dim, int) or self.action_embed_dim < 0:
            raise ValueError("action_embed_dim must be a nonnegative integer")
        if self.modifier_log_scale_bound is not None and not 0 < self.modifier_log_scale_bound <= 8:
            raise ValueError("modifier_log_scale_bound must be in (0, 8]")
        if self.agentmixer_correlation not in ("factorized", "joint"):
            raise ValueError("agentmixer_correlation must be factorized or joint")


def entropy_coefficient(config, steps):
    if config.entropy_final is None:
        return config.entropy_coef
    progress = max(0., (steps / config.steps - config.entropy_decay_start) / (1 - config.entropy_decay_start))
    return config.entropy_coef + min(1., progress) * (config.entropy_final - config.entropy_coef)


def make_policy(config):
    if config.algorithm == "mappo":
        return IndependentPolicy(1, 2, 3, config.hidden_dim)
    if config.algorithm in ("arppo", "bppo"):
        from .bppo import AutoregressivePolicy
        return AutoregressivePolicy(1, 2, 3, config.hidden_dim, action_embed_dim=config.action_embed_dim)
    if config.agentmixer_correlation == "joint":
        from .agentmixer_joint import JointAgentMixerPolicy as AgentMixerPolicy
    else:
        from .agentmixer import AgentMixerPolicy
    return AgentMixerPolicy(1, 1, 2, 3, config.hidden_dim, log_scale_bound=config.modifier_log_scale_bound)


def likelihood(policy, algorithm, obs, state, actions):
    if algorithm == "agentmixer":
        return policy.evaluate_actions(obs, state, actions)
    return policy.evaluate_actions(obs, actions)


@torch.no_grad()
def evaluate(policy, config, game):
    """Enumerate all 9 actions; no Monte Carlo noise or selected episodes."""
    actions = game.joint_actions()
    obs, state = game.observations(len(actions))
    logp, _ = likelihood(policy, config.algorithm, obs, state, actions)
    probability = logp.sum(-1).exp()
    if not torch.isclose(probability.sum(), torch.tensor(1.), atol=1e-5):
        raise RuntimeError("Policy does not define a normalized joint distribution")
    reward = game.reward(actions)
    one_obs, one_state = game.observations(1)
    if config.algorithm == "agentmixer":
        chosen = policy.local_actions(one_obs)
        local_logits = policy.local_logits(obs)
        local_probability = torch.distributions.Categorical(logits=local_logits).log_prob(actions).sum(-1).exp()
    else:
        chosen, _ = policy.sample(one_obs, deterministic=True)
        local_probability = probability
    diagnostics = {}
    if config.algorithm == "agentmixer":
        scales = policy.modifier_log_scales(policy.local_logits(one_obs), one_state).exp()
        diagnostics = {"modifier_scale_min": scales.min().item(), "modifier_scale_max": scales.max().item()}
    return {
        "greedy_return": game.reward(chosen).item(), "greedy_actions": chosen[0].tolist(),
        "training_policy_expected_return": (probability * reward).sum().item(),
        "execution_policy_expected_return": (local_probability * reward).sum().item(),
        "optimal_action_probability": local_probability[reward == game.optimum].sum().item(),
        "training_optimal_action_probability": probability[reward == game.optimum].sum().item(),
        "training_joint_entropy": -(probability * probability.clamp_min(1e-38).log()).sum().item(),
        "execution_joint_entropy": -(local_probability * local_probability.clamp_min(1e-38).log()).sum().item(),
        "training_joint_probabilities": probability.tolist(),
        "execution_joint_probabilities": local_probability.tolist(),
        "optimum_per_step": game.optimum,
        **diagnostics,
    }


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}


def train(config: Config, output: Path):
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.threads)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    game = MatrixGame(config.game)
    policy = make_policy(config)
    critic = mlp(1, 1, config.hidden_dim)
    optimizers = {"critic": torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate, eps=1e-5)}
    if config.algorithm == "bppo":
        for i, actor in enumerate(policy.actors):
            optimizers[f"actor_{i}"] = torch.optim.Adam(actor.parameters(), lr=config.learning_rate, eps=1e-5)
    elif config.algorithm == "agentmixer":
        actor_parameters = list(policy.local_actors.parameters())
        actor_ids = {id(p) for p in actor_parameters}
        modifier_parameters = [p for p in policy.parameters() if id(p) not in actor_ids]
        optimizers["policy"] = torch.optim.Adam([
            {"params": actor_parameters, "lr": config.learning_rate},
            {"params": modifier_parameters, "lr": config.modifier_learning_rate}], eps=1e-5)
    else:
        optimizers["policy"] = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
    metadata = {"config": asdict(config), "python": platform.python_version(), "torch": str(torch.__version__),
                "numpy": np.__version__, "source_sha256": source_hashes(),
                "task_scope": "one-step matrix development experiment; not full paper reproduction",
                "method_variant": {"bppo": "corrected_first_order_peer_surrogate", "agentmixer": (
                    "experimental_joint_gap_scaling" if config.agentmixer_correlation == "joint"
                    else "corrected_mode_preserving_gap_scaling")}.get(config.algorithm, "reference"),
                "evaluation": "exact joint-action enumeration; AgentMixer executes local policies",
                "created_utc": datetime.now(timezone.utc).isoformat()}
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + '\n')
    started = time.monotonic()
    steps, update = 0, 0
    initial = {"steps": 0, "update": 0, **evaluate(policy, config, game)}
    with (output / "metrics.jsonl").open('w') as log:
        log.write(json.dumps(initial) + '\n'); log.flush()
        while steps < config.steps:
            count = min(config.batch_size, config.steps - steps)
            entropy_coef = entropy_coefficient(config, steps)
            obs, state = game.observations(count)
            with torch.no_grad():
                if config.algorithm == "agentmixer":
                    actions, old_log_probs = policy.sample(obs, state)
                else:
                    actions, old_log_probs = policy.sample(obs)
                rewards = game.reward(actions)
                values = critic(state).squeeze(-1)
                advantages = rewards - values
                advantages = normalize_advantages(advantages)
            batch = Batch(obs, state, actions, old_log_probs, advantages, rewards)
            if config.algorithm == "bppo":
                from .bppo import bppo_update
                actor_opts = [optimizers[f"actor_{i}"] for i in range(2)]
                stats = bppo_update(policy, actor_opts, batch.obs, batch.actions, batch.old_log_probs,
                                    batch.advantages, clip_coef=config.clip_coef, entropy_coef=entropy_coef,
                                    epochs=config.epochs, max_grad_norm=config.max_grad_norm, temperature=config.temperature)
            else:
                for _ in range(config.epochs):
                    logp, entropy = likelihood(policy, config.algorithm, obs, state, actions)
                    loss, stats = ppo_objective(logp, old_log_probs, advantages, entropy,
                        clip_coef=config.clip_coef, entropy_coef=entropy_coef, joint=config.algorithm == "agentmixer")
                    optimizers["policy"].zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                    optimizers["policy"].step()
            for _ in range(config.epochs):
                value_loss = .5 * (critic(state).squeeze(-1) - rewards).square().mean()
                optimizers["critic"].zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
                optimizers["critic"].step()
            steps += count; update += 1
            if update % config.eval_every == 0 or steps == config.steps:
                record = {"steps": steps, "update": update, "elapsed_seconds": time.monotonic() - started,
                          "rollout_return": rewards.mean().item(), "value_loss": value_loss.item(),
                          "entropy_coefficient": entropy_coef,
                          **stats, **evaluate(policy, config, game)}
                if not all(np.isfinite(v) for v in record.values() if isinstance(v, (int, float))):
                    raise FloatingPointError("Non-finite training metric")
                log.write(json.dumps(record) + '\n'); log.flush()
                print(json.dumps(record), flush=True)
    save_checkpoint(output / 'checkpoint.pt', policy=policy, critic=critic, optimizers=optimizers, config=asdict(config), step=steps)
    summary = {"status": "completed", "steps": steps, "elapsed_seconds": time.monotonic() - started, **evaluate(policy, config, game)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON containing Config fields')
    parser.add_argument('--algorithm', choices=['mappo','arppo','bppo','agentmixer'])
    parser.add_argument('--game', choices=['climbing','penalty'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    values = json.loads(args.config.read_text()) if args.config else {}
    for key in ['algorithm', 'game', 'seed', 'steps', 'batch_size']:
        if getattr(args, key) is not None: values[key] = getattr(args, key)
    config = Config(**values)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]
    output = args.output or Path('results') / config.algorithm / config.game / f'seed_{config.seed}' / stamp
    print(f'Output: {output}', flush=True)
    train(config, output)

if __name__ == '__main__':
    main()
