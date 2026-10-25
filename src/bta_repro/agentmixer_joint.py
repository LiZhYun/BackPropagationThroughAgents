"""Experimental two-agent AgentMixer with an exactly enumerated joint policy.

Positive joint-action gap scales preserve the Cartesian product of local
mode sets while permitting correlated exploration. This is a named redesign,
not the accepted paper's discrete implementation or a convergence guarantee.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .agentmixer import _MixerBlock, _action_mask


def joint_mode_preserving_logits(
    local_logits: Tensor,
    log_scales: Tensor,
    available_actions: Tensor | None = None,
) -> Tensor:
    """Return joint logits [B,K,K] from local logits [B,2,K].

    E(a,b) = exp(log_scale(a,b)) * (gap_0(a) + gap_1(b)).
    All legal local-mode pairs have energy zero; every other legal pair has
    positive energy. Numerical guards preserve that distinction under extreme
    finite inputs. Gradients through each masked local maximum remain intact.
    """
    if local_logits.ndim != 3 or local_logits.shape[1] != 2 or local_logits.shape[2] < 1:
        raise ValueError("local_logits must have shape [batch, 2, num_actions]")
    batch, _, actions = local_logits.shape
    if log_scales.shape != (batch, actions, actions):
        raise ValueError("log_scales must have shape [batch, num_actions, num_actions]")
    if not local_logits.is_floating_point() or not log_scales.is_floating_point():
        raise ValueError("local logits and log scales must be floating point")
    if local_logits.device != log_scales.device:
        raise ValueError("local logits and log scales must use the same device")
    if not torch.isfinite(local_logits).all() or not torch.isfinite(log_scales).all():
        raise ValueError("local logits and log scales must be finite")
    available = _action_mask(local_logits, available_actions)
    dtype = torch.float64 if local_logits.dtype == torch.float64 else torch.float32
    local = local_logits.to(dtype)
    maximum = local.masked_fill(~available, -torch.inf).amax(dim=-1, keepdim=True)
    gaps = (maximum - torch.where(available, local, maximum)).clamp(0.0, 1e4 * math.exp(8))
    pair_gaps = gaps[:, 0, :, None] + gaps[:, 1, None, :]
    scales = log_scales.to(dtype).clamp(-8.0, 8.0).exp()
    energies = (scales * pair_gaps).clamp(max=1e4)
    minimum_energy = 8 * torch.finfo(dtype).eps * max(1.0, math.log(actions * actions))
    energies = torch.where(pair_gaps > 0, energies.clamp_min(minimum_energy), energies)
    joint_available = available[:, 0, :, None] & available[:, 1, None, :]
    return (-energies).masked_fill(~joint_available, -torch.inf)


class JointPolicyModifier(nn.Module):
    """Mix both local policies and state, then predict one scale per action pair."""

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.policy_embedding = nn.Linear(num_actions, hidden_dim)
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.blocks = nn.Sequential(_MixerBlock(2, 2 * hidden_dim), _MixerBlock(2, 2 * hidden_dim))
        self.scale_head = nn.Linear(4 * hidden_dim, num_actions * num_actions)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, local_logits: Tensor, state: Tensor) -> Tensor:
        policy_features = torch.tanh(self.policy_embedding(local_logits))
        state_features = torch.tanh(self.state_embedding(state)).unsqueeze(1).expand_as(policy_features)
        mixed = self.blocks(torch.cat((policy_features, state_features), dim=-1))
        return self.scale_head(mixed.flatten(1)).reshape(-1, self.num_actions, self.num_actions)


class JointAgentMixerPolicy(nn.Module):
    """Two-agent correlated modifier with exactly evaluated K² joint actions.

    Centralized log probabilities and entropies are [B,1]: one joint event,
    not two per-agent terms. Decentralized sampling returns [B,2] local log
    probabilities. This explicit difference limits current integration to the
    matrix trainer and two-agent Predator–Prey runner. The joint event is
    scored once; it must never be duplicated across agents in PPO.
    """

    joint_event = True

    def __init__(
        self, obs_dim: int, state_dim: int, num_agents: int, num_actions: int,
        hidden_dim: int = 64, log_scale_bound: float | None = None,
        *, distribution_float64: bool = False,
    ) -> None:
        super().__init__()
        if num_agents != 2:
            raise ValueError("the experimental joint modifier supports exactly two agents")
        if min(obs_dim, state_dim, num_actions, hidden_dim) < 1:
            raise ValueError("all policy dimensions must be positive")
        if log_scale_bound is not None and not 0 < log_scale_bound <= 8:
            raise ValueError("log_scale_bound must be in (0, 8]")
        if not isinstance(distribution_float64, bool):
            raise ValueError('distribution_float64 must be boolean')
        self.obs_dim, self.state_dim = obs_dim, state_dim
        self.num_agents, self.num_actions = num_agents, num_actions
        self.log_scale_bound = log_scale_bound
        self.distribution_float64 = distribution_float64
        self.local_actors = nn.ModuleList([
            nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
                          nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                          nn.Linear(hidden_dim, num_actions))
            for _ in range(2)
        ])
        for actor in self.local_actors:
            for layer in actor:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(actor[-1].weight, gain=0.01)
        self.modifier = JointPolicyModifier(state_dim, num_actions, hidden_dim)

    def local_logits(self, obs: Tensor) -> Tensor:
        if obs.ndim != 3 or obs.shape[1:] != (2, self.obs_dim):
            raise ValueError("obs must have shape [batch, 2, obs_dim]")
        return torch.stack([actor(obs[:, i]) for i, actor in enumerate(self.local_actors)], dim=1)

    def modifier_log_scales(self, local: Tensor, state: Tensor) -> Tensor:
        if state is None or state.shape != (local.shape[0], self.state_dim):
            raise ValueError("state must have shape [batch, state_dim]")
        if not torch.isfinite(state).all():
            raise ValueError("state must be finite")
        raw = self.modifier(local, state)
        if self.log_scale_bound is None:
            return raw.clamp(-8.0, 8.0)
        return self.log_scale_bound * torch.tanh(raw / self.log_scale_bound)

    def joint_logits(self, obs: Tensor, state: Tensor, available_actions: Tensor | None = None) -> Tensor:
        local = self.local_logits(obs)
        scales = self.modifier_log_scales(local, state)
        if self.distribution_float64:
            # Keep networks/optimizer in their original dtype. Only the tiny
            # enumerated distribution needs greater separation near local ties.
            local, scales = local.double(), scales.double()
        return joint_mode_preserving_logits(local, scales, available_actions)

    def local_actions(self, obs: Tensor, available_actions: Tensor | None = None) -> Tensor:
        local = self.local_logits(obs)
        return local.masked_fill(~_action_mask(local, available_actions), -torch.inf).argmax(-1)

    def sample(
        self, obs: Tensor, state: Tensor | None = None, available_actions: Tensor | None = None,
        deterministic: bool = False, decentralized: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if decentralized:
            local = self.local_logits(obs)
            local = local.masked_fill(~_action_mask(local, available_actions), -torch.inf)
            distribution = Categorical(logits=local)
            actions = local.argmax(-1) if deterministic else distribution.sample()
            return actions, distribution.log_prob(actions)
        logits = self.joint_logits(obs, state, available_actions)
        distribution = Categorical(logits=logits.flatten(1))
        if deterministic:
            # Ancestral marginal-greedy choices need not be the joint MAP.
            actions = self.local_actions(obs, available_actions)
            indices = actions[:, 0] * self.num_actions + actions[:, 1]
        else:
            indices = distribution.sample()
            actions = torch.stack((indices // self.num_actions, indices % self.num_actions), dim=-1)
        return actions, distribution.log_prob(indices).unsqueeze(-1)

    def evaluate_actions(
        self, obs: Tensor, state: Tensor, actions: Tensor, available_actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        logits = self.joint_logits(obs, state, available_actions).flatten(1)
        if actions.shape != (obs.shape[0], 2) or actions.dtype not in (torch.int32, torch.int64):
            raise ValueError("actions must be integer indices with shape [batch, 2]")
        if actions.device != logits.device or ((actions < 0) | (actions >= self.num_actions)).any():
            raise ValueError("actions must use the policy device and contain valid indices")
        indices = actions[:, 0].long() * self.num_actions + actions[:, 1].long()
        if not torch.isfinite(logits.gather(1, indices.unsqueeze(-1))).all():
            raise ValueError("cannot evaluate an unavailable joint action")
        distribution = Categorical(logits=logits)
        return distribution.log_prob(indices).unsqueeze(-1), distribution.entropy().unsqueeze(-1)
