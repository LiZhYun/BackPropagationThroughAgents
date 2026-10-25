"""Composable synchronous rollouts and updates for feed-forward MARL policies.

Environment adapters own simulator details and physical action clipping. This
module owns rollout boundaries, terminal-observation bootstrapping, and a
single, explicit time-major flattening convention.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .core import Batch, generalized_advantage, normalize_advantages as normalize_team_advantages, ppo_objective
from .value_normalization import ValueNormalizedCritic, validate_value_loss_options, value_regression_loss


@dataclass
class RolloutBatch(Batch):
    """Training fields flatten [time, env] to batch; audit fields retain [T,E]."""

    active_agents: Tensor | None = None
    rewards: Tensor | None = None
    values: Tensor | None = None
    next_values: Tensor | None = None
    terminated: Tensor | None = None
    truncated: Tensor | None = None
    rollout_shape: tuple[int, int] = (0, 0)


class ObservationCarry(list):
    """List of next observations, retaining unfinished episode accounting."""

    def __init__(self, observations, episode_returns, episode_lengths, environment_ids):
        super().__init__(observations)
        self.episode_returns = list(episode_returns)
        self.episode_lengths = list(episode_lengths)
        self.environment_ids = tuple(environment_ids)


def _algorithm(algorithm: str) -> None:
    if algorithm not in {"mappo", "arppo", "bppo", "agentmixer"}:
        raise ValueError("algorithm must be mappo, arppo, bppo, or agentmixer.")


def _spec(env) -> tuple:
    discrete = bool(env.is_discrete)
    dimensions = (env.num_agents, env.obs_dim, env.state_dim,
                  env.num_actions if discrete else env.action_dim)
    if any(not isinstance(d, (int, np.integer)) or d < 1 for d in dimensions):
        raise ValueError("Environment dimensions must be positive integers.")
    return (discrete, *dimensions)


def _model_location(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters(), None)
    return (parameter.device, parameter.dtype) if parameter is not None else (torch.device("cpu"), torch.float32)


def _stack_observations(observations, spec, device, dtype, require_actions=True):
    discrete, agents, obs_dim, state_dim, action_dim = spec
    obs, state, masks, active = [], [], [], []
    for entry in observations:
        local = torch.as_tensor(entry["obs"], device=device, dtype=dtype)
        global_state = torch.as_tensor(entry["state"], device=device, dtype=dtype)
        if local.shape != (agents, obs_dim) or global_state.shape != (state_dim,):
            raise ValueError("Environment obs/state do not match their declared dimensions.")
        if not torch.isfinite(local).all() or not torch.isfinite(global_state).all():
            raise ValueError("Environment obs/state must be finite.")
        alive = torch.as_tensor(entry.get("active_agents", np.ones(agents, dtype=bool)), device=device)
        if alive.shape != (agents,) or not ((alive == 0) | (alive == 1)).all():
            raise ValueError("active_agents must be a binary [agents] mask.")
        obs.append(local)
        state.append(global_state)
        active.append(alive.bool())
        if discrete:
            available = torch.as_tensor(
                entry.get("available_actions", np.ones((agents, action_dim), dtype=bool)), device=device
            )
            if available.shape != (agents, action_dim) or not ((available == 0) | (available == 1)).all():
                raise ValueError("available_actions must be a binary [agents, actions] mask.")
            if require_actions and not available.bool().any(-1).all():
                raise ValueError("Every agent, including inactive agents, needs a legal action.")
            masks.append(available.bool())
    return (torch.stack(obs), torch.stack(state), torch.stack(masks) if discrete else None,
            torch.stack(active))


def _values(critic: nn.Module, state: Tensor) -> Tensor:
    value = critic(state)
    if value.shape == (len(state), 1):
        value = value[:, 0]
    if value.shape != (len(state),) or not torch.isfinite(value).all():
        raise ValueError("The centralized critic must return finite values of shape [batch] or [batch, 1].")
    return value


def _sample(policy, algorithm, is_discrete, obs, state, available):
    if algorithm == "agentmixer":
        return policy.sample(obs, state, available_actions=available) if is_discrete else policy.sample(obs, state)
    return policy.sample(obs, available_actions=available) if is_discrete else policy.sample(obs)


def _likelihood(policy, algorithm, is_discrete, batch):
    if algorithm == "agentmixer":
        if is_discrete:
            return policy.evaluate_actions(batch.obs, batch.state, batch.actions, batch.available_actions)
        return policy.evaluate_actions(batch.obs, batch.state, batch.actions)
    if is_discrete:
        return policy.evaluate_actions(batch.obs, batch.actions, batch.available_actions)
    return policy.evaluate_actions(batch.obs, batch.actions)


@torch.no_grad()
def collect_rollout(
    envs: list,
    policy: nn.Module,
    critic: nn.Module,
    algorithm: str,
    steps: int,
    observations: list[dict] | None = None,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[RolloutBatch, ObservationCarry, list[dict[str, Any]]]:
    """Collect ``steps`` transitions from each explicitly-reset environment.

    Environments must not autoreset. Their terminal/truncated next state is
    evaluated before resetting. Return the observation carry to the next call
    to preserve partial episode returns and lengths across rollout boundaries.
    For reproducible initial resets, pass a list of externally seeded reset
    observations; subsequent resets do not reseed the environments.
    """
    _algorithm(algorithm)
    if not envs or not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("Provide at least one environment and a positive integer step count.")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must lie in [0, 1].")
    spec = _spec(envs[0])
    if any(_spec(env) != spec for env in envs[1:]):
        raise ValueError("Synchronous environments must have matching dimensions and action type.")
    device, dtype = _model_location(policy)
    critic_parameter = next(critic.parameters(), None)
    if critic_parameter is not None and (critic_parameter.device != device or critic_parameter.dtype != dtype):
        raise ValueError("Policy and critic must use the same device and dtype.")
    count = len(envs)
    env_ids = tuple(id(env) for env in envs)
    if observations is None:
        current = [deepcopy(env.reset()[0]) for env in envs]
        episode_returns, episode_lengths = [0.0] * count, [0] * count
    else:
        if len(observations) != count:
            raise ValueError("observations must contain one entry per environment.")
        current = deepcopy(list(observations))
        if isinstance(observations, ObservationCarry):
            if observations.environment_ids != env_ids:
                raise ValueError("Observation carry belongs to a different environment list or order.")
            episode_returns = observations.episode_returns.copy()
            episode_lengths = observations.episode_lengths.copy()
        else:
            episode_returns, episode_lengths = [0.0] * count, [0] * count

    buffers: dict[str, list[Tensor]] = {key: [] for key in (
        "obs", "state", "actions", "old_log_probs", "available_actions", "active_agents",
        "rewards", "values", "next_values", "terminated", "truncated"
    )}
    records = []
    for _ in range(steps):
        obs, state, available, active = _stack_observations(current, spec, device, dtype)
        actions, log_probs = _sample(policy, algorithm, spec[0], obs, state, available)
        expected_actions = (count, spec[1]) if spec[0] else (count, spec[1], spec[4])
        if actions.shape != expected_actions or log_probs.shape != (count, spec[1]):
            raise ValueError("Policy sample returned inconsistent action or log-probability shapes.")
        if not torch.isfinite(actions).all() or not torch.isfinite(log_probs).all():
            raise ValueError("Policy samples and log probabilities must be finite.")
        for key, value in (("obs", obs), ("state", state), ("actions", actions),
                           ("old_log_probs", log_probs), ("active_agents", active)):
            buffers[key].append(value.detach().clone())
        if available is not None:
            buffers["available_actions"].append(available.clone())
        buffers["values"].append(_values(critic, state))

        following, rewards, terminated, truncated, infos = [], [], [], [], []
        for i, env in enumerate(envs):
            # A private numpy copy also protects stored latent Gaussian actions
            # from adapters that clip their input array in place.
            raw_action = actions[i].detach().cpu().numpy().copy()
            next_obs, reward, terminal, timeout, info = env.step(raw_action)
            reward_array = np.asarray(reward)
            if reward_array.ndim != 0 or not np.isfinite(reward_array):
                raise ValueError("Environment reward must be a finite scalar team reward.")
            if not isinstance(terminal, (bool, np.bool_)) or not isinstance(timeout, (bool, np.bool_)):
                raise ValueError("Environment terminated/truncated flags must be boolean.")
            following.append(deepcopy(next_obs))
            rewards.append(float(reward_array))
            terminated.append(bool(terminal))
            truncated.append(bool(timeout))
            infos.append(deepcopy(info))
            episode_returns[i] += float(reward_array)
            episode_lengths[i] += 1

        # These states still belong to the episodes that just stepped. Never
        # substitute a reset state for a truncation bootstrap observation.
        _, next_state, _, _ = _stack_observations(following, spec, device, dtype, require_actions=False)
        buffers["next_values"].append(_values(critic, next_state))
        buffers["rewards"].append(torch.tensor(rewards, device=device, dtype=dtype))
        buffers["terminated"].append(torch.tensor(terminated, device=device, dtype=torch.bool))
        buffers["truncated"].append(torch.tensor(truncated, device=device, dtype=torch.bool))
        current = following
        for i, env in enumerate(envs):
            if terminated[i] or truncated[i]:
                records.append({"env_index": i, "return": episode_returns[i], "length": episode_lengths[i],
                                "terminated": terminated[i], "truncated": truncated[i], "info": infos[i]})
                current[i] = deepcopy(env.reset()[0])
                episode_returns[i], episode_lengths[i] = 0.0, 0

    tensors = {key: torch.stack(values) for key, values in buffers.items() if values}
    advantages, returns = generalized_advantage(
        tensors["rewards"], tensors["values"], tensors["next_values"],
        tensors["terminated"], tensors["truncated"], gamma=gamma, gae_lambda=gae_lambda,
    )
    flatten = lambda value: value.flatten(0, 1)
    batch = RolloutBatch(
        obs=flatten(tensors["obs"]), state=flatten(tensors["state"]),
        actions=flatten(tensors["actions"]), old_log_probs=flatten(tensors["old_log_probs"]),
        advantages=flatten(advantages), returns=flatten(returns),
        available_actions=flatten(tensors["available_actions"]) if spec[0] else None,
        active_agents=flatten(tensors["active_agents"]), rewards=tensors["rewards"],
        values=tensors["values"], next_values=tensors["next_values"],
        terminated=tensors["terminated"], truncated=tensors["truncated"], rollout_shape=(steps, count),
    )
    carry = ObservationCarry(current, episode_returns, episode_lengths, env_ids)
    return batch, carry, records


def update_rollout(
    policy: nn.Module,
    critic: nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    batch: Batch,
    algorithm: str,
    is_discrete: bool,
    *,
    epochs: int = 4,
    clip_coef: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    value_clip_coef: float | None = None,
    value_huber_delta: float | None = None,
    max_grad_norm: float = 0.5,
    normalize_advantages: bool = True,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Update actors, then a centralized scalar critic, on the full rollout.

    Optimizer keys are ``critic`` plus ``actor_0``, ... for BPPO, or ``policy``
    for other methods. Defaults preserve value_coef * raw mean_squared_error.
    A ValueNormalizedCritic updates its moments once from this rollout's raw
    returns; clipping then transforms old raw predictions with those moments.
    """
    _algorithm(algorithm)
    validate_value_loss_options(value_clip_coef, value_huber_delta)
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if not math.isfinite(clip_coef) or not 0 <= clip_coef < 1:
        raise ValueError("clip_coef must lie in [0, 1).")
    if not math.isfinite(entropy_coef) or entropy_coef < 0:
        raise ValueError("entropy_coef must be finite and nonnegative.")
    if not math.isfinite(value_coef) or value_coef < 0:
        raise ValueError("value_coef must be finite and nonnegative.")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive.")
    if batch.obs.ndim != 3 or len(batch.obs) == 0:
        raise ValueError("A nonempty [batch, agents, obs_dim] rollout is required.")
    count, agents = batch.obs.shape[:2]
    if batch.state.ndim != 2 or len(batch.state) != count or batch.old_log_probs.shape != (count, agents):
        raise ValueError("States and old log probabilities must share the rollout batch dimension.")
    if batch.advantages.shape != (count,) or batch.returns.shape != (count,):
        raise ValueError("advantages and returns must have shape [batch].")
    if not torch.isfinite(batch.advantages).all() or not torch.isfinite(batch.returns).all():
        raise ValueError("advantages and returns must be finite.")
    old_values = None
    if value_clip_coef is not None:
        old_values = getattr(batch, "values", None)
        if (old_values is None or old_values.numel() != count
                or not torch.isfinite(old_values).all()):
            raise ValueError("value clipping requires finite old raw rollout values")
        old_values = old_values.detach().flatten()
    active = getattr(batch, "active_agents", None)
    if active is None:
        active = torch.ones((count, agents), dtype=torch.bool, device=batch.obs.device)
    elif active.shape != (count, agents) or not ((active == 0) | (active == 1)).all():
        raise ValueError("active_agents must be a binary [batch, agents] mask.")
    if active.device != batch.obs.device:
        raise ValueError("active_agents and observations must use the same device.")
    active = active.bool()
    if algorithm == "bppo" and not is_discrete and not active.all():
        raise ValueError("Continuous BPPO currently requires every agent to be active.")
    keys = ["critic"] + ([f"actor_{i}" for i in range(agents)] if algorithm == "bppo" else ["policy"])
    if any(key not in optimizers for key in keys):
        raise ValueError(f"Required optimizer keys: {', '.join(keys)}")

    advantages = batch.advantages.detach()
    live = active.any(dim=-1)
    if normalize_advantages:
        advantages = normalize_team_advantages(advantages, active=live)
    if algorithm == "bppo":
        actor_optimizers = [optimizers[f"actor_{i}"] for i in range(agents)]
        kwargs = dict(clip_coef=clip_coef, entropy_coef=entropy_coef, epochs=epochs, max_grad_norm=max_grad_norm)
        if is_discrete:
            from .bppo import bppo_update
            stats = bppo_update(
                policy, actor_optimizers, batch.obs, batch.actions, batch.old_log_probs, advantages,
                available_actions=batch.available_actions, active_agents=active, temperature=temperature, **kwargs,
            )
        else:
            from .bppo_continuous import continuous_bppo_update
            stats = continuous_bppo_update(
                policy, actor_optimizers, batch.obs, batch.actions, batch.old_log_probs, advantages, **kwargs,
            )
    else:
        totals: dict[str, float] = {}
        if live.any():
            for _ in range(epochs):
                log_probs, entropy = _likelihood(policy, algorithm, is_discrete, batch)
                old_log_probs = batch.old_log_probs
                if algorithm == "agentmixer":
                    # Remove inactive factors before forming the joint event ratio.
                    log_probs, entropy = log_probs * active, entropy * active
                    old_log_probs = old_log_probs * active
                loss, metrics = ppo_objective(
                    log_probs, old_log_probs, advantages, entropy, clip_coef=clip_coef,
                    entropy_coef=entropy_coef, joint=algorithm == "agentmixer",
                    active=live if algorithm == "agentmixer" else active,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite policy loss.")
                optimizers["policy"].zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm, error_if_nonfinite=True)
                optimizers["policy"].step()
                metrics["grad_norm"] = float(grad_norm)
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
            stats = {key: value / epochs for key, value in totals.items()}
            stats["num_updates"] = float(epochs)
        else:
            stats = dict.fromkeys(("policy_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm", "num_updates"), 0.0)

    raw_returns = batch.returns.detach()
    normalized = isinstance(critic, ValueNormalizedCritic)
    if normalized:
        critic.update_statistics(raw_returns)
        targets = critic.normalizer.normalize(raw_returns)
        old_predictions = critic.normalizer.normalize(old_values) if old_values is not None else None
    else:
        targets, old_predictions = raw_returns, old_values

    def regression():
        prediction = _values(critic.normalized_forward if normalized else critic, batch.state.detach())
        raw_prediction = critic.normalizer.denormalize(prediction) if normalized else prediction
        raw_mse = (raw_prediction - raw_returns).square().mean()
        regression_loss = value_regression_loss(prediction, targets, old_prediction=old_predictions,
                                                clip_coef=value_clip_coef, huber_delta=value_huber_delta)
        return regression_loss, raw_mse

    value_totals = {"value_loss": 0.0, "value_mse": 0.0, "value_regression_loss": 0.0, "critic_grad_norm": 0.0}
    if value_coef > 0:
        for _ in range(epochs):
            regression_loss, mse = regression()
            value_loss = value_coef * regression_loss
            if not torch.isfinite(value_loss):
                raise FloatingPointError("Non-finite value loss.")
            optimizers["critic"].zero_grad(set_to_none=True)
            value_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizers["critic"].step()
            value_totals["value_loss"] += float(value_loss.detach()) / epochs
            value_totals["value_mse"] += float(mse.detach()) / epochs
            value_totals["value_regression_loss"] += float(regression_loss.detach()) / epochs
            value_totals["critic_grad_norm"] += float(grad_norm) / epochs
    else:
        with torch.no_grad():
            regression_loss, mse = regression()
            value_totals["value_mse"] = float(mse)
            value_totals["value_regression_loss"] = float(regression_loss)
    value_totals.update(
        raw_value_mse=value_totals["value_mse"], value_targets_normalized=float(normalized),
        value_target_mean=float(critic.normalizer.mean) if normalized else 0.0,
        value_target_scale=float(critic.normalizer.scale) if normalized else 1.0,
        value_target_count=float(critic.normalizer.count) if normalized else 0.0,
        value_normalizer_updates=float(critic.normalizer.updates) if normalized else 0.0,
    )
    return {**stats, **value_totals, "value_coef": float(value_coef),
            "critic_updates": float(epochs if value_coef > 0 else 0), "active_fraction": float(active.float().mean())}
