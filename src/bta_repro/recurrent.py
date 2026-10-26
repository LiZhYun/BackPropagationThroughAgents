"""Optional local actor memory with explicit one-decision gradient truncation.

Each actor owns its encoder, GRU cell and action head. Incoming hidden state is
stored beside the raw observation in a rollout, then held fixed during PPO.
The GRU and encoder are recomputed, with gradients, on every policy evaluation.
No network keeps mutable inference state internally.
"""

from functools import partial
import math

import torch
from torch import nn

from .agentmixer import AgentMixerPolicy
from .bppo import AutoregressivePolicy
from .core import IndependentPolicy


class RecurrentActor(nn.Module):
    """Inputs are [local observation, incoming memory, predecessor features]."""

    def __init__(self, input_dim, num_actions, hidden_dim, *, raw_obs_dim):
        super().__init__()
        self.raw_obs_dim, self.memory_dim = raw_obs_dim, hidden_dim
        predecessors = input_dim - raw_obs_dim - hidden_dim
        if min(raw_obs_dim, hidden_dim, num_actions) < 1 or predecessors < 0:
            raise ValueError("invalid recurrent actor dimensions")
        self.input_dim = input_dim
        self.encoder = nn.Linear(raw_obs_dim, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim + predecessors, hidden_dim), nn.Tanh(),
                                  nn.Linear(hidden_dim, num_actions))
        for layer in (self.encoder, self.head[0], self.head[-1]):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.head[-1].weight, gain=.01)
        for name, parameter in self.cell.named_parameters():
            nn.init.zeros_(parameter) if "bias" in name else nn.init.orthogonal_(parameter)

    def memory_step(self, inputs):
        observation = inputs[:, :self.raw_obs_dim]
        incoming = inputs[:, self.raw_obs_dim:self.raw_obs_dim + self.memory_dim].detach()
        return self.cell(torch.tanh(self.encoder(observation)), incoming)

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim or not torch.isfinite(inputs).all():
            raise ValueError("invalid recurrent actor input")
        memory = self.memory_step(inputs)
        predecessors = inputs[:, self.raw_obs_dim + self.memory_dim:]
        return self.head(torch.cat((memory, predecessors), dim=-1))


class _LocalMemory:
    """Only local observation history enters memory; current actions enter the head."""

    def pack_observations(self, observation, memory=None):
        if observation.ndim != 3 or observation.shape[1:] != (self.num_agents, self.raw_obs_dim):
            raise ValueError("raw recurrent observations must be [batch, agents, raw_obs_dim]")
        if not observation.is_floating_point() or not torch.isfinite(observation).all():
            raise ValueError("raw recurrent observations must be finite floating-point values")
        shape = (*observation.shape[:2], self.memory_dim)
        if memory is None:
            memory = observation.new_zeros(shape)
        if (not isinstance(memory, torch.Tensor) or memory.shape != shape
                or memory.device != observation.device or memory.dtype != observation.dtype
                or not torch.isfinite(memory).all()):
            raise ValueError("incoming memory must match observation batch, agents, device and dtype")
        return torch.cat((observation, memory.detach()), dim=-1)

    @torch.no_grad()
    def advance_memory(self, packed_observation):
        if (packed_observation.ndim != 3 or packed_observation.shape[1:] != (self.num_agents, self.obs_dim)
                or not packed_observation.is_floating_point() or not torch.isfinite(packed_observation).all()):
            raise ValueError("packed recurrent observation shape mismatch")
        actors = self.local_actors if isinstance(self, AgentMixerPolicy) else self.actors
        # Projected autoregressive actors wrap the action head in .network.
        modules = [actor if isinstance(actor, RecurrentActor) else actor.network for actor in actors]
        return torch.stack([actor.memory_step(packed_observation[:, i]) for i, actor in enumerate(modules)], dim=1)


class RecurrentIndependentPolicy(_LocalMemory, IndependentPolicy):
    def __init__(self, obs_dim, num_agents, num_actions, hidden_dim=64):
        self.raw_obs_dim, self.memory_dim, self.obs_dim = obs_dim, hidden_dim, obs_dim + hidden_dim
        super().__init__(self.obs_dim, num_agents, num_actions, hidden_dim,
                         actor_factory=partial(RecurrentActor, raw_obs_dim=obs_dim))


class RecurrentAutoregressivePolicy(_LocalMemory, AutoregressivePolicy):
    def __init__(self, obs_dim, num_agents, num_actions, hidden_dim=64, action_embed_dim=0):
        self.raw_obs_dim, self.memory_dim = obs_dim, hidden_dim
        super().__init__(obs_dim + hidden_dim, num_agents, num_actions, hidden_dim, action_embed_dim,
                         actor_factory=partial(RecurrentActor, raw_obs_dim=obs_dim))


class RecurrentAgentMixerPolicy(_LocalMemory, AgentMixerPolicy):
    def __init__(self, obs_dim, state_dim, num_agents, num_actions, hidden_dim=64, log_scale_bound=None):
        self.raw_obs_dim, self.memory_dim = obs_dim, hidden_dim
        super().__init__(obs_dim + hidden_dim, state_dim, num_agents, num_actions, hidden_dim, log_scale_bound,
                         actor_factory=partial(RecurrentActor, raw_obs_dim=obs_dim))
