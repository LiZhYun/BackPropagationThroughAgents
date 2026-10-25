"""Continuous BPPO with raw diagonal-Gaussian autoregressive actions.

An action is one vector event: coordinate log probabilities and entropies are
summed before PPO ratios are formed. Environment clipping belongs to the
adapter and must not overwrite the raw actions stored in the rollout.
"""

from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal

from .bppo import _actor_objective, _successor_feedback
from .core import LOG_STD_MAX, LOG_STD_MIN, initial_log_std


class _GaussianActor(nn.Module):
    """Tanh mean network and independent, bounded, state-independent scales."""

    def __init__(
        self, input_dim: int, action_dim: int, hidden_dim: int, *, initial_std: float = 1.0
    ) -> None:
        super().__init__()
        log_std = initial_log_std(initial_std)
        self.mean = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        for layer in self.mean:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean[-1].weight, gain=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std))

    def forward(self, inputs: Tensor) -> Independent:
        mean = self.mean(inputs)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mean)
        # Reinterpret the final coordinate axis as one action event. Every
        # likelihood and entropy used below consequently has shape [batch].
        return Independent(Normal(mean, std), 1)


class ContinuousAutoregressivePolicy(nn.Module):
    """Separate Gaussian actors conditioned on the raw predecessor actions."""

    def __init__(
        self, obs_dim: int, num_agents: int, action_dim: int, hidden_dim: int = 64,
        *, initial_std: float = 1.0,
    ) -> None:
        super().__init__()
        if min(obs_dim, num_agents, action_dim, hidden_dim) < 1:
            raise ValueError("All policy dimensions must be positive.")
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.actors = nn.ModuleList(
            _GaussianActor(obs_dim + i * action_dim, action_dim, hidden_dim, initial_std=initial_std)
            for i in range(num_agents)
        )

    def _validate(self, obs: Tensor, actions: Tensor | None = None) -> None:
        if obs.ndim != 3 or obs.shape[1:] != (self.num_agents, self.obs_dim):
            raise ValueError("obs must have shape [batch, num_agents, obs_dim].")
        if obs.shape[0] == 0:
            raise ValueError("The batch must not be empty.")
        if not obs.is_floating_point() or not torch.isfinite(obs).all():
            raise ValueError("obs must contain finite floating-point values.")
        if actions is not None:
            if actions.shape != (*obs.shape[:2], self.action_dim):
                raise ValueError("actions must have shape [batch, agents, action_dim].")
            if not actions.is_floating_point() or not torch.isfinite(actions).all():
                raise ValueError("actions must contain finite floating-point raw actions.")
            if actions.device != obs.device or actions.dtype != obs.dtype:
                raise ValueError("actions and obs must have the same device and dtype.")

    def _distribution(
        self,
        obs: Tensor,
        actions: Tensor,
        agent: int,
        available_actions: Tensor | None = None,
    ) -> Independent:
        # The optional argument lets the generic BPPO successor derivative
        # work for both discrete features and raw continuous action vectors.
        if available_actions is not None:
            raise ValueError("Continuous policies do not support discrete availability masks.")
        inputs = torch.cat((obs[:, agent], actions[:, :agent].flatten(1)), dim=-1)
        return self.actors[agent](inputs)

    @torch.no_grad()
    def sample(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        self._validate(obs)
        actions = obs.new_zeros((obs.shape[0], self.num_agents, self.action_dim))
        log_probs = []
        for i in range(self.num_agents):
            distribution = self._distribution(obs, actions, i)
            action = distribution.mean if deterministic else distribution.sample()
            actions[:, i] = action
            log_probs.append(distribution.log_prob(action))
        return actions, torch.stack(log_probs, dim=1)

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        self._validate(obs, actions)
        distributions = [self._distribution(obs, actions, i) for i in range(self.num_agents)]
        return (
            torch.stack([d.log_prob(actions[:, i]) for i, d in enumerate(distributions)], dim=1),
            torch.stack([d.entropy() for d in distributions], dim=1),
        )


def continuous_bppo_update(
    policy: ContinuousAutoregressivePolicy,
    optimizers: list[torch.optim.Optimizer],
    obs: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    *,
    clip_coef: float = 0.2,
    entropy_coef: float = 0.01,
    epochs: int = 4,
    max_grad_norm: float = 0.5,
    peer_noise: str = "fresh",
) -> dict[str, float]:
    """Reverse-order BPPO on one fresh rollout, using raw Gaussian rsamples.

    The corrected scalar PPO loss, peer dot product, and full numerator and
    denominator action derivative are shared with discrete BPPO. Actors and
    their optimizer states are independent. Advantages are not normalized here.
    ``peer_noise='behavior'`` reconstructs and holds the recorded action's
    Gaussian noise fixed across epochs; ``'fresh'`` retains the original control.
    """
    policy._validate(obs, actions)
    if (
        old_log_probs.shape != obs.shape[:2]
        or not old_log_probs.is_floating_point()
        or not torch.isfinite(old_log_probs).all()
    ):
        raise ValueError("old_log_probs must be finite floats with shape [batch, agents].")
    if advantages.shape != (obs.shape[0],) or not torch.isfinite(advantages).all():
        raise ValueError("advantages must be finite with shape [batch].")
    if old_log_probs.device != obs.device or advantages.device != obs.device:
        raise ValueError("Rollout tensors must be on the same device.")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if not math.isfinite(clip_coef) or not 0 <= clip_coef < 1:
        raise ValueError("clip_coef must be in [0, 1).")
    if not math.isfinite(entropy_coef) or entropy_coef < 0:
        raise ValueError("entropy_coef must be finite and nonnegative.")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive.")
    if peer_noise not in ("fresh", "behavior"):
        raise ValueError("peer_noise must be fresh or behavior.")
    if len(optimizers) != policy.num_agents:
        raise ValueError("Provide exactly one optimizer per actor.")
    for actor, optimizer in zip(policy.actors, optimizers):
        actor_ids = {id(p) for p in actor.parameters()}
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if actor_ids != optimizer_ids:
            raise ValueError("Each optimizer must own exactly its corresponding actor's parameters.")

    obs, actions = obs.detach(), actions.detach()
    advantages, old_log_probs = advantages.detach(), old_log_probs.detach()
    behavior = deepcopy(policy).requires_grad_(False)
    with torch.no_grad():
        behavior_log_probs, _ = behavior.evaluate_actions(obs, actions)
        if not torch.allclose(behavior_log_probs, old_log_probs, atol=1e-5, rtol=1e-5):
            raise ValueError("Policy changed since rollout; the behavior snapshot would be invalid.")

    totals: dict[str, float] = {}
    num_updates = 0
    for i in reversed(range(policy.num_agents)):
        multiplier, coefficients = _successor_feedback(
            policy, behavior, obs, actions, actions, i
        )
        if peer_noise == "behavior" and i < policy.num_agents - 1:
            old_distribution = behavior._distribution(obs, actions, i).base_dist
            # Recover the exogenous noise of the raw rollout sample. The
            # behavior policy and recorded predecessor inputs stay fixed.
            noise = ((actions[:, i] - old_distribution.loc) / old_distribution.scale).detach()
        for _ in range(epochs):
            distribution = policy._distribution(obs, actions, i)
            if i == policy.num_agents - 1:
                reparameterized_action = torch.zeros_like(actions[:, i])
            elif peer_noise == "behavior":
                base = distribution.base_dist
                reparameterized_action = base.loc + base.scale * noise
            else:
                reparameterized_action = distribution.rsample()
            loss, metrics = _actor_objective(
                distribution.log_prob(actions[:, i]),
                distribution.entropy(),
                old_log_probs[:, i],
                advantages,
                multiplier,
                coefficients,
                reparameterized_action,
                clip_coef=clip_coef,
                entropy_coef=entropy_coef,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite continuous BPPO loss.")
            optimizers[i].zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                policy.actors[i].parameters(), max_grad_norm, error_if_nonfinite=True
            )
            optimizers[i].step()
            metrics["grad_norm"] = grad_norm.detach()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            num_updates += 1
    result = {key: value / num_updates for key, value in totals.items()}
    result["num_updates"] = float(num_updates)
    result["peer_noise_behavior"] = float(peer_noise == "behavior")
    return result
