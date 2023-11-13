import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bta.algorithms.utils.util import init, check
from bta.algorithms.utils.cnn import CNNBase
from bta.algorithms.utils.mlp import MLPBase, MLPLayer
from bta.algorithms.utils.mix import MIXBase
from bta.algorithms.utils.rnn import RNNLayer
from bta.algorithms.utils.act import ACTLayer
from bta.algorithms.utils.gobigger.encoder import Encoder
from bta.algorithms.utils.popart import PopArt
from bta.utils.util import get_shape_from_obs_space, read_config, deep_merge_dicts

class QMixer(nn.Module):
    def __init__(self, args, share_obs_space, device=torch.device("cpu")):
        super(QMixer, self).__init__()

        self.args = args
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
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        self.num_agents = args.num_agents
        self.state_dim = get_shape_from_obs_space(share_obs_space)[0]
        self.state_dim = self.state_dim

        self.embed_dim = args.hidden_size

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.num_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        agent_qs = check(agent_qs).to(**self.tpdv)
        states = check(states).to(**self.tpdv)
        # noise = check(noise).to(**self.tpdv)
        
        bs = agent_qs.size(0) // self.num_agents
        # states = torch.cat([states, noise], dim=-1)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.num_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.num_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Skip connections
        s = 0
        # Compute final output
        y = torch.bmm(hidden, w_final) + v + s
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
