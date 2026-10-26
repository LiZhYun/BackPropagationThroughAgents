"""Discrete autoregressive BPPO with explicit own and peer gradient terms.

The corrected clipping convention and behavior-denominator derivative are
documented in ``docs/bppo.md``. No simulator dependency is needed here.
"""

from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F


def _actor_network(input_dim: int, num_actions: int, hidden_dim: int) -> nn.Sequential:
    """Preserve the original network construction and initialization order."""
    network = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, num_actions),
    )
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(network[-1].weight, gain=0.01)
    return network


class _ProjectedConditionalActor(nn.Module):
    """A successor owns one action projection, shared across its input slots."""

    def __init__(self, obs_dim, predecessors, num_actions, hidden_dim, action_embed_dim, actor_factory=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.predecessors = predecessors
        self.num_actions = num_actions
        self.action_projection = nn.Linear(num_actions, action_embed_dim, bias=False)
        nn.init.orthogonal_(self.action_projection.weight, gain=1.0)
        factory = _actor_network if actor_factory is None else actor_factory
        self.network = factory(obs_dim + predecessors * action_embed_dim, num_actions, hidden_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        preceding_actions = inputs[:, self.obs_dim :].reshape(
            len(inputs), self.predecessors, self.num_actions
        )
        # Project inside the actor, retaining action-input derivatives for BPPO.
        projected = self.action_projection(preceding_actions).flatten(1)
        return self.network(torch.cat((inputs[:, : self.obs_dim], projected), dim=-1))


class AutoregressivePolicy(nn.Module):
    """Separate actors, with actor i conditioned on actions 0 through i-1."""

    def __init__(
        self, obs_dim: int, num_agents: int, num_actions: int, hidden_dim: int = 64,
        action_embed_dim: int = 0, *, actor_factory=None,
    ) -> None:
        super().__init__()
        if min(obs_dim, num_agents, num_actions, hidden_dim) < 1:
            raise ValueError("All policy dimensions must be positive.")
        if not isinstance(action_embed_dim, int) or isinstance(action_embed_dim, bool) or action_embed_dim < 0:
            raise ValueError("action_embed_dim must be a nonnegative integer.")
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.action_embed_dim = action_embed_dim
        self.actors = nn.ModuleList()
        factory = _actor_network if actor_factory is None else actor_factory
        for i in range(num_agents):
            actor = (
                _ProjectedConditionalActor(obs_dim, i, num_actions, hidden_dim, action_embed_dim, actor_factory)
                if action_embed_dim > 0 and i > 0
                else factory(obs_dim + i * num_actions, num_actions, hidden_dim)
            )
            self.actors.append(actor)

    def _validate(
        self, obs: Tensor, actions: Tensor | None, available_actions: Tensor | None
    ) -> None:
        if obs.ndim != 3 or obs.shape[1:] != (self.num_agents, self.obs_dim):
            raise ValueError("obs must have shape [batch, num_agents, obs_dim].")
        if obs.shape[0] == 0:
            raise ValueError("The batch must not be empty.")
        if not obs.is_floating_point() or not torch.isfinite(obs).all():
            raise ValueError("obs must contain finite floating-point values.")
        if available_actions is not None:
            if available_actions.shape != (*obs.shape[:2], self.num_actions):
                raise ValueError("available_actions must have shape [batch, agents, actions].")
            if available_actions.device != obs.device:
                raise ValueError("available_actions and obs must be on the same device.")
            if not ((available_actions == 0) | (available_actions == 1)).all():
                raise ValueError("available_actions must be a boolean or binary mask.")
            if not available_actions.bool().any(dim=-1).all():
                raise ValueError("Every agent must have at least one available action.")
        if actions is not None:
            if actions.shape != obs.shape[:2] or actions.dtype != torch.long:
                raise ValueError("actions must be a long tensor of shape [batch, agents].")
            if actions.device != obs.device:
                raise ValueError("actions and obs must be on the same device.")
            if ((actions < 0) | (actions >= self.num_actions)).any():
                raise ValueError("actions contain an out-of-range action.")
            if available_actions is not None:
                legal = available_actions.bool().gather(-1, actions.unsqueeze(-1))
                if not legal.all():
                    raise ValueError("actions contain an unavailable action.")

    def _distribution(
        self,
        obs: Tensor,
        action_features: Tensor,
        agent: int,
        available_actions: Tensor | None,
    ) -> Categorical:
        inputs = torch.cat((obs[:, agent], action_features[:, :agent].flatten(1)), dim=-1)
        logits = self.actors[agent](inputs)
        if available_actions is not None:
            logits = logits.masked_fill(~available_actions[:, agent].bool(), -torch.inf)
        return Categorical(logits=logits)

    @torch.no_grad()
    def sample(
        self,
        obs: Tensor,
        available_actions: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor]:
        self._validate(obs, None, available_actions)
        features = obs.new_zeros((obs.shape[0], self.num_agents, self.num_actions))
        actions, log_probs = [], []
        for i in range(self.num_agents):
            distribution = self._distribution(obs, features, i, available_actions)
            action = distribution.logits.argmax(-1) if deterministic else distribution.sample()
            features[:, i] = F.one_hot(action, self.num_actions).to(obs.dtype)
            actions.append(action)
            log_probs.append(distribution.log_prob(action))
        return torch.stack(actions, dim=1), torch.stack(log_probs, dim=1)

    def evaluate_actions(
        self, obs: Tensor, actions: Tensor, available_actions: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        self._validate(obs, actions, available_actions)
        features = F.one_hot(actions, self.num_actions).to(obs.dtype)
        distributions = [
            self._distribution(obs, features, i, available_actions)
            for i in range(self.num_agents)
        ]
        return (
            torch.stack([d.log_prob(actions[:, i]) for i, d in enumerate(distributions)], 1),
            torch.stack([d.entropy() for d in distributions], 1),
        )


def _successor_feedback(
    policy: AutoregressivePolicy,
    behavior: AutoregressivePolicy,
    obs: Tensor,
    actions: Tensor,
    features: Tensor,
    agent: int,
    available_actions: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return detached M and dM/da_i at the recorded joint action.

    Both the current and the frozen behavior conditional likelihoods are
    differentiated with respect to the action probe. Behavior *parameters*
    are frozen; the denominator is not constant as a function of that probe.
    """
    if agent == policy.num_agents - 1:
        return obs.new_ones(obs.shape[0]), torch.zeros_like(features[:, agent])
    probe = features[:, agent].detach().requires_grad_(True)
    probed_features = torch.cat(
        (features[:, :agent], probe.unsqueeze(1), features[:, agent + 1 :]), dim=1
    )
    log_ratio = obs.new_zeros(obs.shape[0])
    for successor in range(agent + 1, policy.num_agents):
        current = policy._distribution(obs, probed_features, successor, available_actions)
        old = behavior._distribution(obs, probed_features, successor, available_actions)
        log_ratio = log_ratio + current.log_prob(actions[:, successor]) - old.log_prob(
            actions[:, successor]
        )
    multiplier = log_ratio.exp()
    coefficients = torch.autograd.grad(multiplier.sum(), probe)[0]
    if not torch.isfinite(multiplier).all() or not torch.isfinite(coefficients).all():
        raise FloatingPointError("Non-finite BPPO successor importance ratio or derivative.")
    return multiplier.detach(), coefficients.detach()


def _actor_objective(
    log_probs: Tensor,
    entropy: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    successor_ratio: Tensor,
    peer_coefficients: Tensor,
    relaxed_action: Tensor,
    *,
    clip_coef: float,
    entropy_coef: float,
    active: Tensor | None = None,
    return_components: bool = False,
) -> tuple[Tensor, dict[str, Tensor]] | tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    """Reduce scalar PPO and action-vector peer terms separately, then add."""
    if active is not None:
        if active.shape != log_probs.shape or not ((active == 0) | (active == 1)).all():
            raise ValueError("active must be a binary mask matching the scalar log probabilities.")
        weights = active.to(log_probs)
    else:
        weights = torch.ones_like(log_probs)
    count = weights.sum().clamp_min(1)

    def reduce(value: Tensor) -> Tensor:
        return (value * weights).sum() / count

    advantages = advantages.detach()
    log_ratio = log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    own_loss = -reduce(successor_ratio.detach() * torch.minimum(unclipped, clipped))

    # Keep the paper's detached ratio on the peer pathway. Its coefficient
    # comes from the branch chosen by the scalar PPO surrogate.
    selected_ratio = torch.where(unclipped <= clipped, ratio, clipped_ratio).detach()
    peer_dot = (peer_coefficients.detach() * relaxed_action).sum(dim=-1)
    peer_loss = -reduce(selected_ratio * advantages * peer_dot)
    mean_entropy = reduce(entropy)
    loss = own_loss + peer_loss - entropy_coef * mean_entropy
    metrics = {
        "loss": loss.detach(),
        "own_loss": own_loss.detach(),
        "peer_loss": peer_loss.detach(),
        "entropy": mean_entropy.detach(),
        "approx_kl": reduce((ratio - 1.0) - log_ratio).detach(),
        "clip_fraction": reduce(((ratio - 1.0).abs() > clip_coef).float()).detach(),
        "successor_ratio": reduce(successor_ratio).detach(),
        "successor_ratio_abs_change": reduce((successor_ratio - 1.0).abs()).detach(),
        "peer_coefficient_norm": reduce(peer_coefficients.norm(dim=-1)).detach(),
    }
    if return_components:
        return loss, metrics, {"own": own_loss, "peer": peer_loss}
    return loss, metrics


def _gradient_diagnostics(components: dict[str, Tensor], parameters: tuple[Tensor, ...]) -> dict[str, Tensor]:
    """Measure pre-clipping component gradients without populating .grad."""
    def gradients(loss):
        if not loss.requires_grad:
            return (None,) * len(parameters)
        return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)

    own, peer = gradients(components["own"]), gradients(components["peer"])
    zero = components["own"].new_zeros(())
    own_norm = sum((g.detach().square().sum() for g in own if g is not None), zero).sqrt()
    peer_norm = sum((g.detach().square().sum() for g in peer if g is not None), zero).sqrt()
    dot = sum((a.detach().mul(b.detach()).sum() for a, b in zip(own, peer)
               if a is not None and b is not None), zero)
    denominator = own_norm * peer_norm
    cosine = torch.where(denominator > 0, dot / denominator.clamp_min(torch.finfo(dot.dtype).tiny), zero)
    return {"own_grad_norm": own_norm, "peer_grad_norm": peer_norm,
            "own_peer_grad_cosine": cosine.clamp(-1.0, 1.0)}


def bppo_update(
    policy: AutoregressivePolicy,
    optimizers: list[torch.optim.Optimizer],
    obs: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    *,
    available_actions: Tensor | None = None,
    active_agents: Tensor | None = None,
    clip_coef: float = 0.2,
    entropy_coef: float = 0.01,
    epochs: int = 4,
    max_grad_norm: float = 0.5,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Update each actor in reverse order, completing its epochs before moving on.

    ``policy`` must still be the rollout policy on entry: a frozen snapshot is
    needed to differentiate the behavior denominator. ``advantages`` is one
    team advantage per transition and is not normalized inside this function.
    Each optimizer must own exactly the parameters of its corresponding actor.
    """
    policy._validate(obs, actions, available_actions)
    if active_agents is not None:
        if active_agents.shape != actions.shape or not ((active_agents == 0) | (active_agents == 1)).all():
            raise ValueError("active_agents must be a binary mask with shape [batch, agents].")
        if active_agents.device != obs.device:
            raise ValueError("active_agents and obs must be on the same device.")
    if old_log_probs.shape != actions.shape or not torch.isfinite(old_log_probs).all():
        raise ValueError("old_log_probs must be finite with shape [batch, agents].")
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
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive.")
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
        behavior_log_probs, _ = behavior.evaluate_actions(obs, actions, available_actions)
        if not torch.allclose(behavior_log_probs, old_log_probs, atol=1e-5, rtol=1e-5):
            raise ValueError("Policy changed since rollout; the behavior snapshot would be invalid.")
    features = F.one_hot(actions, policy.num_actions).to(obs.dtype)
    totals: dict[str, float] = {}
    num_updates = 0
    for i in reversed(range(policy.num_agents)):
        active = None if active_agents is None else active_agents[:, i]
        if active is not None and not active.any():
            # Do not advance optimizer state for an actor absent from this batch.
            continue
        # Successors have finished updating and remain fixed during actor i's epochs.
        multiplier, coefficients = _successor_feedback(
            policy, behavior, obs, actions, features, i, available_actions
        )
        for _ in range(epochs):
            distribution = policy._distribution(obs, features, i, available_actions)
            if i < policy.num_agents - 1:
                relaxed_action = F.gumbel_softmax(
                    distribution.logits, tau=temperature, hard=True, dim=-1
                )
            else:
                relaxed_action = torch.zeros_like(distribution.logits)
            loss, metrics, components = _actor_objective(
                distribution.log_prob(actions[:, i]),
                distribution.entropy(),
                old_log_probs[:, i],
                advantages,
                multiplier,
                coefficients,
                relaxed_action,
                clip_coef=clip_coef,
                entropy_coef=entropy_coef,
                active=active,
                return_components=True,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite BPPO loss.")
            metrics.update(_gradient_diagnostics(components, tuple(policy.actors[i].parameters())))
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
    if num_updates == 0:
        totals = dict.fromkeys(
            ("loss", "own_loss", "peer_loss", "entropy", "approx_kl", "clip_fraction",
             "successor_ratio", "successor_ratio_abs_change", "peer_coefficient_norm", "grad_norm",
             "own_grad_norm", "peer_grad_norm", "own_peer_grad_cosine"), 0.0
        )
    result = {key: value / max(num_updates, 1) for key, value in totals.items()}
    result["num_updates"] = float(num_updates)
    return result
