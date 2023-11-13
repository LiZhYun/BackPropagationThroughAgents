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

class R_Critic(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
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

        input_shape = get_shape_from_obs_space(obs_space)[0]

        if action_space.__class__.__name__ == "Discrete":
            self.discrete_action = True
            self.action_dim = 1
        elif action_space.__class__.__name__ == "Box":
            self.continuous_action = True
            self.action_dim = action_space.shape[0]

        self.fc1 = nn.Linear(input_shape, args.hidden_size)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        self.fc2 = nn.Linear(args.hidden_size+self.action_dim, self.action_dim)
        self.dep_fc1 = nn.Linear(self.action_dim * args.num_agents, self.hidden_size)
        self.dep_fc2 = nn.Linear(self.hidden_size + args.hidden_size, self.action_dim)

        self.to(device)

    def forward(self, obs, rnn_states, action, masks, parents_actions, execution_mask, dep_mode, task_id=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        parents_actions = check(parents_actions).to(**self.tpdv)
        execution_mask = check(execution_mask).to(**self.tpdv)

        x = F.relu(self.fc1(obs))
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            x, rnn_states = self.rnn(x, rnn_states, masks)
        q = self.fc2(torch.cat([x, action], dim=-1))

        dep_in = self.dep_fc1(parents_actions * execution_mask.unsqueeze(-1))
        dep_final = torch.cat([x, dep_in], dim=-1)

        if dep_mode:
            dep = self.dep_fc2(dep_final)
            q = q.detach() + dep

        return q, rnn_states
