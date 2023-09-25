import torch
import torch.nn as nn
import torch.nn.functional as F
from bta.algorithms.utils.util import init, check
from bta.algorithms.utils.cnn import CNNBase
from bta.algorithms.utils.mlp import MLPBase, MLPLayer
from bta.algorithms.utils.mix import MIXBase
from bta.algorithms.utils.rnn import RNNLayer
from bta.algorithms.utils.act import ACTLayer
from bta.algorithms.utils.gobigger.encoder import Encoder
from bta.algorithms.utils.popart import PopArt
from bta.utils.util import get_shape_from_obs_space, read_config, deep_merge_dicts

class RNNAgent(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(RNNAgent, self).__init__()
        self.args = args
        self.num_agents = args.num_agents
        self.noise_dim = args.noise_dim
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal  
        self._activation_id = args.activation_id     
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._use_popart = args.use_popart
        self._influence_layer_N = args.influence_layer_N
        self._recurrent_N = args.recurrent_N
        self._num_v_out = getattr(args, "num_v_out", 1)
        self.tpdv = dict(dtype=torch.float32, device=device)

        input_shape = get_shape_from_obs_space(obs_space)[0]

        if action_space.__class__.__name__ == "Discrete":
            self.discrete_action = True
            self.action_dim = action_space.n
        elif action_space.__class__.__name__ == "Box":
            self.continuous_action = True
            self.action_dim = action_space.shape[0]

        self.fc1 = nn.Linear(input_shape, args.hidden_size)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        self.fc2 = nn.Linear(args.hidden_size, self.action_dim)

        self.noise_fc1 = nn.Linear(args.noise_dim + args.num_agents, args.hidden_size)
        self.noise_fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.noise_fc3 = nn.Linear(args.hidden_size, self.action_dim)

        self.hyper = True
        self.hyper_noise_fc1 = nn.Linear(args.noise_dim + args.num_agents, args.hidden_size * self.action_dim)

    def forward(self, obs, rnn_states, masks, noise, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        noise = check(noise).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        noise = noise.reshape(-1, self.noise_dim)
        agent_ids = torch.eye(self.args.num_agents).repeat(noise.shape[0]//self.num_agents, 1).to(**self.tpdv)

        x = F.relu(self.fc1(obs))
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            x, rnn_states = self.rnn(x, rnn_states, masks)
        q = self.fc2(x)

        noise_input = torch.cat([noise, agent_ids], dim=-1)

        if self.hyper:
            W = self.hyper_noise_fc1(noise_input).reshape(-1, self.action_dim, self.hidden_size)
            wq = torch.bmm(W, x.unsqueeze(2)).squeeze(-1)
        else:
            z = F.tanh(self.noise_fc1(noise_input))
            z = F.tanh(self.noise_fc2(z))
            wz = self.noise_fc3(z)

            wq = q * wz

        return wq, rnn_states
