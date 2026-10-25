"""Continuous AgentMixer with identical local and centralized Gaussian means.

Actions are unconstrained Normal samples. The environment adapter may clip
actuator commands, but PPO must store and score the original raw samples.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from bta_repro.agentmixer import _MixerBlock


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class _GaussianActor(nn.Module):
    """One agent's observation-conditioned mean and diagonal uncertainty."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        for layer in self.features:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        features = self.features(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std.exp()


class ContinuousPolicyModifier(nn.Module):
    """Mix local Gaussian parameters and state to adjust log standard deviations."""

    def __init__(self, state_dim: int, num_agents: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.policy_embedding = nn.Linear(2 * action_dim, hidden_dim)
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.blocks = nn.Sequential(
            _MixerBlock(num_agents, 2 * hidden_dim),
            _MixerBlock(num_agents, 2 * hidden_dim),
        )
        self.log_std_head = nn.Linear(2 * hidden_dim, action_dim)
        # A zero residual starts the centralized policy at the local policy.
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, mean: Tensor, std: Tensor, state: Tensor) -> Tensor:
        policy_features = torch.tanh(self.policy_embedding(torch.cat((mean, std), dim=-1)))
        state_features = torch.tanh(self.state_embedding(state)).unsqueeze(-2)
        state_features = state_features.expand_as(policy_features)
        mixed = self.blocks(torch.cat((policy_features, state_features), dim=-1))
        return self.log_std_head(mixed)


class ContinuousAgentMixerPolicy(nn.Module):
    """Diagonal Gaussian AgentMixer with decentralized mean execution.

    Observation shape is [batch, agents, obs_dim], state shape is
    [batch, state_dim], and raw action shape is [batch, agents, action_dim].
    Log probabilities and entropies sum action coordinates, returning one
    value per agent. The centralized mean is exactly the local actor mean.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        num_agents: int,
        action_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if min(obs_dim, state_dim, num_agents, action_dim, hidden_dim) < 1:
            raise ValueError("all policy dimensions must be positive")
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.local_actors = nn.ModuleList([
            _GaussianActor(obs_dim, action_dim, hidden_dim) for _ in range(num_agents)
        ])
        self.modifier = ContinuousPolicyModifier(state_dim, num_agents, action_dim, hidden_dim)

    def local_parameters(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        if obs.ndim != 3 or obs.shape[1:] != (self.num_agents, self.obs_dim):
            raise ValueError("obs must have shape [batch, num_agents, obs_dim]")
        if not torch.isfinite(obs).all():
            raise ValueError("obs must be finite")
        parameters = [actor(obs[:, i]) for i, actor in enumerate(self.local_actors)]
        mean = torch.stack([entry[0] for entry in parameters], dim=1)
        std = torch.stack([entry[1] for entry in parameters], dim=1)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("local Gaussian parameters must be finite")
        return mean, std

    def joint_parameters(self, obs: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        mean, local_std = self.local_parameters(obs)
        if state is None or state.shape != (obs.shape[0], self.state_dim):
            raise ValueError("state must have shape [batch, state_dim]")
        if not torch.isfinite(state).all():
            raise ValueError("state must be finite")
        residual = self.modifier(mean, local_std, state)
        if not torch.isfinite(residual).all():
            raise ValueError("modifier log standard deviations must be finite")
        joint_log_std = (local_std.log() + residual).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, joint_log_std.exp()

    def local_actions(self, obs: Tensor) -> Tensor:
        """Return the local Gaussian modes without consulting global state."""
        return self.local_parameters(obs)[0]

    def sample(
        self,
        obs: Tensor,
        state: Tensor | None = None,
        deterministic: bool = False,
        decentralized: bool = False,
    ) -> tuple[Tensor, Tensor]:
        mean, std = self.local_parameters(obs) if decentralized else self.joint_parameters(obs, state)
        distribution = Normal(mean, std)
        raw_actions = mean if deterministic else distribution.sample()
        return raw_actions, distribution.log_prob(raw_actions).sum(dim=-1)

    def evaluate_actions(
        self, obs: Tensor, state: Tensor, raw_actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        mean, std = self.joint_parameters(obs, state)
        if raw_actions.shape != mean.shape or not raw_actions.is_floating_point():
            raise ValueError("raw_actions must be floats with shape [batch, num_agents, action_dim]")
        if not torch.isfinite(raw_actions).all():
            raise ValueError("raw_actions must be finite")
        distribution = Normal(mean, std)
        return distribution.log_prob(raw_actions).sum(-1), distribution.entropy().sum(-1)
