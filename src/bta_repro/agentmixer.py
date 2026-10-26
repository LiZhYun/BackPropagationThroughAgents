"""AgentMixer: centralized exploration with decentralized, mode-consistent actors.

The discrete modifier deliberately corrects the paper's unrestricted additive
Gumbel offsets, which do not preserve local modes. See ``docs/agentmixer.md``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


def _action_mask(logits: Tensor, available_actions: Tensor | None) -> Tensor:
    if available_actions is None:
        return torch.ones_like(logits, dtype=torch.bool)
    if available_actions.shape != logits.shape:
        raise ValueError("available_actions must have the same shape as logits")
    if available_actions.device != logits.device:
        raise ValueError("available_actions and logits must be on the same device")
    if not torch.all((available_actions == 0) | (available_actions == 1)):
        raise ValueError("available_actions must contain only zero/one or booleans")
    mask = available_actions.bool()
    if not mask.any(dim=-1).all():
        raise ValueError("each agent must have at least one available action")
    return mask


def mode_preserving_logits(
    local_logits: Tensor,
    log_scales: Tensor,
    available_actions: Tensor | None = None,
) -> Tensor:
    """Scale each action's gap from its agent's best *available* local logit.

    For a masked maximum ``m`` and positive scales ``s``, return
    ``-s * (m - local_logits)``. Every original mode remains exactly zero;
    every available nonmode is negative. Masks remain negative infinity.
    Scales and gaps are bounded for numerical safety; gradients through the
    maximum are retained. Half precision inputs are calculated in float32.
    """
    if local_logits.ndim < 2 or local_logits.shape != log_scales.shape:
        raise ValueError("local_logits and log_scales must have equal [..., actions] shapes")
    if local_logits.shape[-1] == 0:
        raise ValueError("at least one action is required")
    if not local_logits.is_floating_point() or not log_scales.is_floating_point():
        raise ValueError("logits and log_scales must be floating point")
    if not torch.isfinite(local_logits).all() or not torch.isfinite(log_scales).all():
        raise ValueError("logits and log_scales must be finite")
    mask = _action_mask(local_logits, available_actions)
    dtype = torch.float64 if local_logits.dtype == torch.float64 else torch.float32
    logits = local_logits.to(dtype)
    scales = log_scales.to(dtype).clamp(-8.0, 8.0).exp()
    maximum = logits.masked_fill(~mask, -torch.inf).amax(dim=-1, keepdim=True)
    # Avoid infinities from masked entries and from extreme finite differences.
    gaps = (maximum - torch.where(mask, logits, maximum)).clamp(0.0, 1e4 * math.exp(8))
    scaled_gaps = (scales * gaps).clamp(max=1e4)
    # A subnormal gap times a small scale can round to zero, introducing a mode.
    # This floor also keeps the distinction after categorical normalization.
    minimum_gap = 8 * torch.finfo(dtype).eps
    scaled_gaps = torch.where(gaps > 0, scaled_gaps.clamp_min(minimum_gap), scaled_gaps)
    return (-scaled_gaps).masked_fill(~mask, -torch.inf)


class _MixerBlock(nn.Module):
    """Alternate communication across agents and mixing within each agent."""

    def __init__(self, num_agents: int, channels: int) -> None:
        super().__init__()
        self.agent_norm = nn.LayerNorm(channels)
        self.agent_mlp = nn.Sequential(
            nn.Linear(num_agents, max(16, 2 * num_agents)),
            nn.GELU(),
            nn.Linear(max(16, 2 * num_agents), num_agents),
        )
        self.channel_norm = nn.LayerNorm(channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, 2 * channels), nn.GELU(), nn.Linear(2 * channels, channels)
        )

    def forward(self, features: Tensor) -> Tensor:
        across_agents = self.agent_mlp(self.agent_norm(features).transpose(-1, -2))
        features = features + across_agents.transpose(-1, -2)
        return features + self.channel_mlp(self.channel_norm(features))


class PolicyModifier(nn.Module):
    """State and local policy parameters determine per-action log gap scales."""

    def __init__(self, state_dim: int, num_agents: int, num_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.policy_embedding = nn.Linear(num_actions, hidden_dim)
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.blocks = nn.Sequential(
            _MixerBlock(num_agents, 2 * hidden_dim),
            _MixerBlock(num_agents, 2 * hidden_dim),
        )
        self.scale_head = nn.Linear(2 * hidden_dim, num_actions)
        # exp(0) = 1: the initial categorical policy equals the local policy,
        # up to the documented guard for extreme or nearly tied logits.
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, local_logits: Tensor, state: Tensor) -> Tensor:
        local_features = torch.tanh(self.policy_embedding(local_logits))
        state_features = torch.tanh(self.state_embedding(state))
        state_features = state_features.unsqueeze(-2).expand_as(local_features)
        features = torch.cat((local_features, state_features), dim=-1)
        return self.scale_head(self.blocks(features))


class AgentMixerPolicy(nn.Module):
    """Discrete AgentMixer policy with independent local actor parameters.

    Observations have shape [batch, agents, obs_dim], states [batch, state_dim],
    and masks [batch, agents, actions]. Sampling is conditionally factorized;
    the modifier's shared information does not create correlated random draws.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        num_agents: int,
        num_actions: int,
        hidden_dim: int = 64,
        log_scale_bound: float | None = None,
        *, actor_factory=None,
    ) -> None:
        super().__init__()
        if min(obs_dim, state_dim, num_agents, num_actions, hidden_dim) < 1:
            raise ValueError("all policy dimensions must be positive")
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.num_actions = num_actions
        if log_scale_bound is not None and not 0 < log_scale_bound <= 8:
            raise ValueError("log_scale_bound must be in (0, 8]")
        self.log_scale_bound = log_scale_bound
        self.local_actors = nn.ModuleList([
            actor_factory(obs_dim, num_actions, hidden_dim) if actor_factory is not None else nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_actions),
            )
            for _ in range(num_agents)
        ])
        for actor in self.local_actors if actor_factory is None else ():
            for layer in actor:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(actor[-1].weight, gain=0.01)
        self.modifier = PolicyModifier(state_dim, num_agents, num_actions, hidden_dim)

    def local_logits(self, obs: Tensor) -> Tensor:
        if obs.ndim != 3 or obs.shape[1:] != (self.num_agents, self.obs_dim):
            raise ValueError("obs must have shape [batch, num_agents, obs_dim]")
        return torch.stack([actor(obs[:, i]) for i, actor in enumerate(self.local_actors)], dim=1)

    def joint_logits(
        self, obs: Tensor, state: Tensor, available_actions: Tensor | None = None
    ) -> Tensor:
        local = self.local_logits(obs)
        if state is None or state.shape != (obs.shape[0], self.state_dim):
            raise ValueError("state must have shape [batch, state_dim]")
        if not torch.isfinite(state).all():
            raise ValueError("state must be finite")
        return mode_preserving_logits(local, self.modifier_log_scales(local, state), available_actions)

    def modifier_log_scales(self, local: Tensor, state: Tensor) -> Tensor:
        """Optional smooth exploration control; identity and local modes remain intact."""
        raw = self.modifier(local, state)
        if self.log_scale_bound is None:
            return raw.clamp(-8., 8.)
        return self.log_scale_bound * torch.tanh(raw / self.log_scale_bound)

    def local_actions(self, obs: Tensor, available_actions: Tensor | None = None) -> Tensor:
        """Execute greedily using only each agent's local observation and mask."""
        logits = self.local_logits(obs)
        return logits.masked_fill(~_action_mask(logits, available_actions), -torch.inf).argmax(-1)

    def sample(
        self,
        obs: Tensor,
        state: Tensor | None,
        available_actions: Tensor | None = None,
        deterministic: bool = False,
        decentralized: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if decentralized:
            logits = self.local_logits(obs)
            logits = logits.masked_fill(~_action_mask(logits, available_actions), -torch.inf)
        else:
            logits = self.joint_logits(obs, state, available_actions)
        distribution = Categorical(logits=logits)
        actions = logits.argmax(-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions)

    def evaluate_actions(
        self,
        obs: Tensor,
        state: Tensor,
        actions: Tensor,
        available_actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        logits = self.joint_logits(obs, state, available_actions)
        if actions.shape != obs.shape[:2] or actions.dtype not in (torch.int32, torch.int64):
            raise ValueError("actions must be integer indices with shape [batch, num_agents]")
        if ((actions < 0) | (actions >= self.num_actions)).any():
            raise ValueError("action index out of range")
        if not torch.isfinite(logits.gather(-1, actions.long().unsqueeze(-1))).all():
            raise ValueError("cannot evaluate an unavailable action")
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy()
