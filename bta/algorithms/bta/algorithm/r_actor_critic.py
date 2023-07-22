import math
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from bta.algorithms.utils.util import init, check
from bta.algorithms.utils.cnn import CNNBase
from bta.algorithms.utils.mlp import MLPBase, MLPLayer
from bta.algorithms.utils.mix import MIXBase
from bta.algorithms.utils.rnn import RNNLayer
from bta.algorithms.utils.act import ACTLayer
from bta.algorithms.utils.popart import PopArt
from bta.algorithms.utils.gobigger.encoder import Encoder
from bta.utils.util import get_shape_from_obs_space, read_config, deep_merge_dicts

class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, agent_id, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal 
        self._activation_id = args.activation_id
        self._use_policy_active_masks = args.use_policy_active_masks 
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._influence_layer_N = args.influence_layer_N 
        self._use_policy_vhead = args.use_policy_vhead
        self._use_popart = args.use_popart 
        self._recurrent_N = args.recurrent_N 
        self.use_action_attention = args.use_action_attention
        self.agent_id = agent_id 
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.whole_cfg = None

        if args.env_name == "GoBigger":
            self._mixed_obs = False
            self._nested_obs = True
            default_config = read_config(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'utils', 'gobigger', 'default_model_config.yaml'))
            config = read_config(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'utils', 'gobigger', 'default_ppo_config.yaml'))
            self.whole_cfg = deep_merge_dicts(default_config, config)
            self.base = Encoder(self.whole_cfg, args)
        elif 'Dict' in obs_shape.__class__.__name__:
            self._mixed_obs = True
            self._nested_obs = False
            self.base = MIXBase(args, obs_shape, cnn_layers_params=args.cnn_layers_params)
        else:
            self._mixed_obs = False
            self._nested_obs = False
            self.base = CNNBase(args, obs_shape, cnn_layers_params=args.cnn_layers_params) if len(obs_shape)==3 \
                else MLPBase(args, obs_shape, use_attn_internal=args.use_attn_internal, use_cat_self=True)
        
        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(obs_shape[0], self.hidden_size,
                              self._influence_layer_N, self._use_orthogonal, self._activation_id)
            input_size += self.hidden_size

        self.abs_size = input_size

        if action_space.__class__.__name__ == "Discrete":
            self.action_dim = action_space.n
        elif action_space.__class__.__name__ == "Box":
            self.action_dim = action_space.shape[0]
        else:
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_dim = continous_dim + discrete_dim

        # input_size += args.num_agents * self.action_dim + args.num_agents
        if self.use_action_attention:
            self.action_base = MLPBase(args, [args.num_agents], use_attn_internal=args.use_attn_internal, use_cat_self=True)
        else:
            self.action_base = MLPBase(args, [args.num_agents * self.action_dim + args.num_agents], use_attn_internal=args.use_attn_internal, use_cat_self=True)
        self.feature_norm = nn.LayerNorm(input_size)

        self.num_agents = args.num_agents

        self.act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, onehot_action, execution_mask, available_actions=None, deterministic=False, tau=1.0):        
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
        masks = check(masks).to(**self.tpdv)
        onehot_action = check(onehot_action).to(**self.tpdv)
        execution_mask = check(execution_mask).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self._nested_obs:
            actor_features = torch.stack([self.base(obs[batch_idx]) for batch_idx in range(obs.shape[0])])
        else:
            actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        masked_actions = (onehot_action * execution_mask.unsqueeze(-1)).view(*onehot_action.shape[:-2], -1)
        # actor_features = torch.cat([actor_features, masked_actions.view(*masked_actions.shape[:-2], -1)], dim=1)

        id_feat = torch.eye(self.args.num_agents)[self.agent_id].unsqueeze(0).repeat(actor_features.shape[0], 1).to(actor_features.device)
        
        if self.use_action_attention:
            actor_features = actor_features + self.action_base(id_feat)
        else:
            actor_features = actor_features + self.action_base(torch.cat([masked_actions, id_feat], dim=1))
        actor_features = self.feature_norm(actor_features)
        obs_feat = actor_features.clone()

        if deterministic:
            logits = None
            actions, action_log_probs, dist_entropy = self.act(actor_features, available_actions, deterministic, tau=tau)
        else:
            actions, action_log_probs, dist_entropy, logits = self.act(actor_features, available_actions, deterministic, tau=tau)
        
        return actions, action_log_probs, rnn_states, logits, dist_entropy, obs_feat
    
    def evaluate_actions(self, obs, rnn_states, action, masks, onehot_action, execution_mask, available_actions=None, active_masks=None, tau=1.0, kl=False, joint_actions=None):
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
        onehot_action = check(onehot_action).to(**self.tpdv)
        execution_mask = check(execution_mask).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if joint_actions is not None:
            joint_actions = check(joint_actions).to(**self.tpdv)
        
        if self._nested_obs:
            actor_features = torch.stack([self.base(obs[batch_idx]) for batch_idx in range(obs.shape[0])])
        else:
            actor_features = self.base(obs)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        masked_actions = (onehot_action * execution_mask.unsqueeze(-1)).view(*onehot_action.shape[:-2], -1)
        # actor_features = torch.cat([actor_features, masked_actions.view(*masked_actions.shape[:-2], -1)], dim=1)

        id_feat = torch.eye(self.args.num_agents)[self.agent_id].unsqueeze(0).repeat(actor_features.shape[0], 1).to(actor_features.device)
        if self.use_action_attention:
            actor_features = actor_features + self.action_base(id_feat)
        else:
            actor_features = actor_features + self.action_base(torch.cat([masked_actions, id_feat], dim=1))
        actor_features = self.feature_norm(actor_features)
        obs_feat = actor_features.clone()

        # actor_features = torch.cat([actor_features, id_feat], dim=1)
        train_actions, action_log_probs, action_log_probs_kl, dist_entropy, logits = self.act.evaluate_actions(actor_features, action, available_actions, active_masks = active_masks if self._use_policy_active_masks else None, rsample=True, tau=tau, kl=kl, joint_actions=joint_actions)
        
        return train_actions, action_log_probs, action_log_probs_kl, dist_entropy, logits, obs_feat

class R_Critic(nn.Module):
    def __init__(self, args, share_obs_space, action_space, device=torch.device("cpu")):
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
        self.num_agents = args.num_agents
        self._num_v_out = getattr(args, "num_v_out", 1)
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        self.whole_cfg = None

        if args.env_name == "GoBigger":
            self._mixed_obs = False
            self._nested_obs = True
            default_config = read_config(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'utils', 'gobigger', 'default_model_config.yaml'))
            config = read_config(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'utils', 'gobigger', 'default_ppo_config.yaml'))
            self.whole_cfg = deep_merge_dicts(default_config, config)
            self.base = Encoder(self.whole_cfg, args)
        elif 'Dict' in share_obs_shape.__class__.__name__:
            self._mixed_obs = True
            self._nested_obs = False
            self.base = MIXBase(args, share_obs_shape, cnn_layers_params=args.cnn_layers_params)
        else:
            self._mixed_obs = False
            self._nested_obs = False
            self.base = CNNBase(args, share_obs_shape, cnn_layers_params=args.cnn_layers_params) if len(share_obs_shape)==3 \
                else MLPBase(args, share_obs_shape, use_attn_internal=True, use_cat_self=args.use_cat_self)

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(share_obs_shape[0], self.hidden_size,
                              self._influence_layer_N, self._use_orthogonal, self._activation_id)
            input_size += self.hidden_size
        
        if args.env_name == "StarCraft2v2":
            input_size += args.num_agents

        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(input_size, self._num_v_out, device=device))
        else:
            self.v_out = init_(nn.Linear(input_size, self._num_v_out))

        self.to(device)

    def forward(self, share_obs, rnn_states, masks, active_masks=None):
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
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if self._nested_obs:
            critic_features = torch.stack([self.base(share_obs[batch_idx]) for batch_idx in range(share_obs.shape[0])])
        else:
            critic_features = self.base(share_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_share_obs = self.mlp(share_obs)
            critic_features = torch.cat([critic_features, mlp_share_obs], dim=1)

        if active_masks is not None:
            critic_features = torch.cat([critic_features, active_masks.reshape(-1, self.num_agents)], dim=1)
        
        values = self.v_out(critic_features)

        return values, rnn_states
