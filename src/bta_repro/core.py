"""Shared numerical primitives. Algorithm-specific objectives live elsewhere."""
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Independent, Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def initial_log_std(initial_std: float) -> float:
    """Validate an initial Gaussian scale against the existing log-scale bounds."""
    try:
        value = float(initial_std)
    except (TypeError, ValueError):
        raise ValueError("initial_std must be finite and within [exp(-5), exp(2)]") from None
    if (isinstance(initial_std, bool) or not math.isfinite(value)
            or not math.exp(LOG_STD_MIN) <= value <= math.exp(LOG_STD_MAX)):
        raise ValueError("initial_std must be finite and within [exp(-5), exp(2)]")
    return math.log(value)


def mlp(input_dim: int, output_dim: int, hidden_dim: int = 64) -> nn.Sequential:
    model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                          nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                          nn.Linear(hidden_dim, output_dim))
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=2 ** .5)
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(model[-1].weight, gain=.01)
    return model


def masked_categorical(logits: Tensor, available_actions: Tensor | None = None) -> Categorical:
    if available_actions is not None:
        if available_actions.shape != logits.shape:
            raise ValueError("available_actions must match logits")
        mask = available_actions.bool()
        if not mask.any(-1).all():
            raise ValueError("every agent must have a valid action")
        logits = logits.masked_fill(~mask, -torch.inf)
    return Categorical(logits=logits)


class IndependentPolicy(nn.Module):
    """MAPPO actor reference: separate local actors and simultaneous PPO updates."""
    def __init__(self, obs_dim, num_agents, num_actions, hidden_dim=64, *, actor_factory=None):
        super().__init__()
        self.num_agents, self.num_actions = num_agents, num_actions
        factory = mlp if actor_factory is None else actor_factory
        self.actors = nn.ModuleList([factory(obs_dim, num_actions, hidden_dim) for _ in range(num_agents)])

    def logits(self, obs):
        return torch.stack([actor(obs[:, i]) for i, actor in enumerate(self.actors)], dim=1)

    def sample(self, obs, available_actions=None, deterministic=False):
        dist = masked_categorical(self.logits(obs), available_actions)
        actions = dist.probs.argmax(-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions)

    def evaluate_actions(self, obs, actions, available_actions=None):
        dist = masked_categorical(self.logits(obs), available_actions)
        return dist.log_prob(actions), dist.entropy()


class ContinuousIndependentPolicy(nn.Module):
    """Local Gaussian actors for the continuous MAPPO reference.

    Each agent's action vector is one probability event. Actions remain raw
    Gaussian samples in the buffer; actuator clipping is the adapter's job.
    """
    def __init__(self, obs_dim, num_agents, action_dim, hidden_dim=64, *, initial_std: float = 1.0):
        super().__init__()
        if min(obs_dim, num_agents, action_dim, hidden_dim) < 1:
            raise ValueError("policy dimensions must be positive")
        log_std = initial_log_std(initial_std)
        self.obs_dim, self.num_agents, self.action_dim = obs_dim, num_agents, action_dim
        self.actors = nn.ModuleList([mlp(obs_dim, action_dim, hidden_dim)
                                     for _ in range(num_agents)])
        self.log_std = nn.Parameter(torch.full((num_agents, action_dim), log_std))

    def distribution(self, obs):
        if obs.ndim != 3 or obs.shape[1:] != (self.num_agents, self.obs_dim):
            raise ValueError("obs must have shape [batch, agents, obs_dim]")
        mean = torch.stack([actor(obs[:, i]) for i, actor in enumerate(self.actors)], dim=1)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mean)
        return Independent(Normal(mean, std), 1)

    @torch.no_grad()
    def sample(self, obs, deterministic=False):
        dist = self.distribution(obs)
        actions = dist.mean if deterministic else dist.sample()
        return actions, dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        if actions.shape != (*obs.shape[:2], self.action_dim):
            raise ValueError("actions must have shape [batch, agents, action_dim]")
        dist = self.distribution(obs)
        return dist.log_prob(actions), dist.entropy()

    def local_actions(self, obs):
        return self.distribution(obs).mean


@dataclass
class Batch:
    """Named on-policy samples avoid positional minibatch tuple mismatches."""
    obs: Tensor
    state: Tensor
    actions: Tensor
    old_log_probs: Tensor
    advantages: Tensor
    returns: Tensor
    available_actions: Tensor | None = None


def normalize_advantages(advantages: Tensor, active: Tensor | None = None) -> Tensor:
    """Normalize valid samples with stable moments; constant signals become zero.

    Float32 means of identical values can differ from those values by a rounding
    unit. Dividing that residual by a tiny standard deviation creates a false
    learning signal. Accumulate both moments in float64 before converting back.
    """
    mask = torch.ones_like(advantages, dtype=torch.bool) if active is None else active.bool()
    if mask.shape != advantages.shape:
        raise ValueError("active mask must match advantages")
    result = torch.zeros_like(advantages)
    if not mask.any():
        return result
    values = advantages[mask].double()
    if not torch.isfinite(values).all():
        raise ValueError("advantages must be finite")
    centered = values - values.mean()
    scale = centered.square().mean().sqrt().clamp_min(1e-8)
    result[mask] = (centered / scale).to(advantages.dtype)
    return result


def generalized_advantage(rewards, values, next_values, terminated, truncated, gamma=.99, gae_lambda=.95):
    """GAE for [time, environment] tensors, with explicit terminal observations.

    next_values must evaluate the observation before any automatic reset.
    Truncation bootstraps that value but stops recursion into the next episode.
    True termination both stops recursion and removes the bootstrap.
    """
    shape = rewards.shape
    if len(shape) != 2 or any(x.shape != shape for x in (values, next_values, terminated, truncated)):
        raise ValueError("GAE inputs must share [time, environment] shape")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("termination and truncation masks must be boolean")
    delta = rewards + gamma * (~terminated) * next_values - values
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(shape[0])):
        carry = delta[t] + gamma * gae_lambda * (~(terminated[t] | truncated[t])) * carry
        advantage[t] = carry
    return advantage, advantage + values


def ppo_objective(log_probs, old_log_probs, advantages, entropy, *, clip_coef=.2,
                  entropy_coef=.01, joint=False, active=None):
    """Clipped PPO with an explicit choice of per-agent or joint likelihood.

    Independent/AR baselines use per-agent ratios; the centralized AgentMixer
    reference uses the product distribution's joint ratio. Never sum scalar
    score terms over action coordinates.
    """
    if log_probs.shape != old_log_probs.shape or entropy.shape != log_probs.shape:
        raise ValueError("log probabilities and entropy must share [batch, agents] shape")
    if advantages.shape != log_probs.shape[:1]:
        raise ValueError("advantages must have shape [batch]")
    log_ratio = log_probs - old_log_probs.detach()
    if joint:
        log_ratio, entropy = log_ratio.sum(-1), entropy.sum(-1)
        advantage = advantages.detach()
    else:
        advantage = advantages.detach().unsqueeze(-1)
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_coef, 1 + clip_coef) * advantage)
    if active is None:
        weights = torch.ones_like(surrogate)
    else:
        weights = active.to(surrogate).expand_as(surrogate)
    count = weights.sum()
    if count.item() == 0:
        return log_probs.sum() * 0., {"policy_loss": 0., "entropy": 0., "approx_kl": 0., "clip_fraction": 0.}
    policy_loss = -(surrogate * weights).sum() / count
    entropy_mean = (entropy * weights).sum() / count
    loss = policy_loss - entropy_coef * entropy_mean
    stats = {"policy_loss": policy_loss.detach().item(), "entropy": entropy_mean.detach().item(),
             "approx_kl": (((ratio - 1) - log_ratio) * weights).sum().detach().item() / count.item(),
             "clip_fraction": (((ratio - 1).abs() > clip_coef) * weights).sum().detach().item() / count.item()}
    return loss, stats
