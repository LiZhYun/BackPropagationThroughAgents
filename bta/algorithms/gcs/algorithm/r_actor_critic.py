import torch
import os
import torch.nn as nn
from bta.algorithms.utils.util import init, check
from bta.algorithms.utils.cnn import CNNBase
from bta.algorithms.utils.mlp import MLPBase
from bta.algorithms.utils.rnn import RNNLayer
from bta.algorithms.utils.act_graph import ACTLayer
from bta.algorithms.utils.gobigger.encoder import Encoder
# from bta.algorithms.utils.act import ACTLayer
from bta.utils.util import get_shape_from_obs_space, read_config, deep_merge_dicts


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.args = args

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.whole_cfg = None

        if args.env_name == "GoBigger":
            self._mixed_obs = False
            self._nested_obs = True
            default_config = read_config(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'utils', 'gobigger', 'default_model_config.yaml'))
            config = read_config(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'utils', 'gobigger', 'default_ppo_config.yaml'))
            self.whole_cfg = deep_merge_dicts(default_config, config)
            self.base = Encoder(self.whole_cfg)
        else:
            self._mixed_obs = False
            self._nested_obs = False
            self.base = CNNBase(args, obs_shape, cnn_layers_params=args.cnn_layers_params) if len(obs_shape)==3 \
                else MLPBase(args, obs_shape, use_attn_internal=args.use_attn_internal, use_cat_self=True)
        
        if args.env_name == "GoBigger":
            input_size = self.base.output_size * args.num_agents
            self.feature_size = self.base.output_size * args.num_agents
        else:
            input_size = self.base.output_size
            self.feature_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size
            self.feature_size = self.hidden_size

        if action_space.__class__.__name__ == "Discrete":
            self.action_dim = action_space.n
        elif action_space.__class__.__name__ == "Box":
            self.action_dim = action_space.shape[0]
        else:
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_dim = continous_dim + discrete_dim

        input_size += self.args.num_agents * self.action_dim
        self.act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain, device)

        self.to(device)

    def forward(self, obs, rnn_states, masks, G_s, available_actions=None, deterministic=False, one_hot_actions=None):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        if self._nested_obs:
            for batch_idx in range(obs.shape[0]):
                for key in obs[batch_idx].keys():
                    if 'Dict' in obs[batch_idx][key].__class__.__name__.capitalize():
                        for sub_key in obs[batch_idx][key].keys():
                            obs[batch_idx][key][sub_key] = check(obs[batch_idx][key][sub_key]).to(**self.tpdv)
                    else:
                        obs[batch_idx][key] = check(obs[batch_idx][key]).to(**self.tpdv)
        elif self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)    # 4.1.64
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            if "Graph" in G_s.__class__.__name__ :
                available_actions = available_actions.reshape(self.args.n_rollout_threads, -1, available_actions.shape[-1])

        if self._nested_obs:
            actor_features = torch.stack([self.base(obs[batch_idx]) for batch_idx in range(obs.shape[0])])
        else:
            actor_features = self.base(obs)

        if(self.args.act_graph):
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                actor_features = actor_features.reshape(-1, self.args.hidden_size)
                rnn_states = rnn_states.reshape(-1, 1, self.args.hidden_size)
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)  # 4.64    4.1.64
            if "list" in G_s.__class__.__name__ :
                actor_features = actor_features.reshape(-1, self.args.num_agents, self.feature_size)
            actions, action_log_probs, father_actions = self.act(obs, actor_features, G_s, available_actions, deterministic, one_hot_actions)
        else:
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)  # 4.64    4.1.64
            actions, action_log_probs = self.act(obs, actor_features, available_actions, deterministic)
        # (4.1)  (4.1)
        return actions, action_log_probs, rnn_states, father_actions

    def evaluate_actions(self, obs, father_action, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self._nested_obs:
            for batch_idx in range(obs.shape[0]):
                for key in obs[batch_idx].keys():
                    if 'Dict' in obs[batch_idx][key].__class__.__name__.capitalize():
                        for sub_key in obs[batch_idx][key].keys():
                            obs[batch_idx][key][sub_key] = check(obs[batch_idx][key][sub_key]).to(**self.tpdv)
                    else:
                        obs[batch_idx][key] = check(obs[batch_idx][key]).to(**self.tpdv)
        elif self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if self._nested_obs:
            actor_features = torch.stack([self.base(obs[batch_idx]) for batch_idx in range(obs.shape[0])])
        else:
            actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.args.env_name == "GRFootball" or self.args.env_name == "gaussian":
            action = action[:, [0]]
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   father_action,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)
        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        self.args = args

        share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.whole_cfg = None

        if args.env_name == "GoBigger":
            self._mixed_obs = False
            self._nested_obs = True
            default_config = read_config(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'utils', 'gobigger', 'default_model_config.yaml'))
            config = read_config(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'utils', 'gobigger', 'default_ppo_config.yaml'))
            self.whole_cfg = deep_merge_dicts(default_config, config)
            self.base = Encoder(self.whole_cfg)
        else:
            self._mixed_obs = False
            self._nested_obs = False
            self.base = CNNBase(args, share_obs_shape, cnn_layers_params=args.cnn_layers_params) if len(share_obs_shape)==3 \
                else MLPBase(args, share_obs_shape, use_attn_internal=True, use_cat_self=args.use_cat_self)

        if args.env_name == "GoBigger":
            input_size = self.base.output_size * args.num_agents
        else:
            input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(input_size, 1))

        self.to(device)

    def forward(self, share_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        if self._nested_obs:
            for batch_idx in range(share_obs.shape[0]):
                for key in share_obs[batch_idx].keys():
                    if 'Dict' in share_obs[batch_idx][key].__class__.__name__.capitalize():
                        for sub_key in share_obs[batch_idx][key].keys():
                            share_obs[batch_idx][key][sub_key] = check(share_obs[batch_idx][key][sub_key]).to(**self.tpdv)
                    else:
                        share_obs[batch_idx][key] = check(share_obs[batch_idx][key]).to(**self.tpdv)
        elif self._mixed_obs:
            for key in share_obs.keys():
                share_obs[key] = check(share_obs[key]).to(**self.tpdv)
        else:
            share_obs = check(share_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)  # 4.1.64
        masks = check(masks).to(**self.tpdv)

        if self._nested_obs:
            critic_features = torch.stack([self.base(share_obs[batch_idx]) for batch_idx in range(share_obs.shape[0])])
        else:
            critic_features = self.base(share_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
