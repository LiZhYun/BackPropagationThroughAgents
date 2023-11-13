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
from bta.algorithms.utils.distributions import DiagGaussian, Categorical, FixedCategorical, FixedNormal
from bta.utils.util import get_shape_from_obs_space, read_config, deep_merge_dicts

class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
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
        
        self.discrete = False
        if action_space.__class__.__name__ == "Discrete":
            self.discrete = True
            self.action_dim = action_space.n
            self.action_shape = 1
            self.act = Categorical(input_size, self.action_dim, self._use_orthogonal, self._gain)
            self.dep_act = Categorical(input_size+self.hidden_size, self.action_dim, self._use_orthogonal, self._gain)
        elif action_space.__class__.__name__ == "Box":
            self.action_dim = action_space.shape[0]
            self.action_shape = action_space.shape[0]
            self.act = DiagGaussian(input_size, self.action_dim, self._use_orthogonal, self._gain)
            self.dep_act = DiagGaussian(input_size+self.hidden_size, self.action_dim, self._use_orthogonal, self._gain)

        # self.act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        if self._use_policy_vhead:
            if self._use_popart:
                self.v_out = init_(PopArt(input_size, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(input_size, 1))
        
        self.dep_fc = nn.Sequential(init_(nn.Linear(args.num_agents * self.action_shape, self.hidden_size)), 
                                           nn.ReLU(),
                                           nn.LayerNorm(self.hidden_size))
        # self.dep_act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain)
        
        # in Overcooked, predict shaped info
        self._predict_other_shaped_info = False
        if args.env_name == "Overcooked" and getattr(args, "predict_other_shaped_info", False):
            self._predict_other_shaped_info = True
            self.pred_shaped_info = init_(nn.Linear(input_size, 12))

        self.to(device)

    def forward(self, obs, rnn_states, masks, parents_actions, execution_mask, dep_mode, available_actions=None, deterministic=False):        
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

        if parents_actions is not None:
            parents_actions = check(parents_actions).to(**self.tpdv)
        if execution_mask is not None:
            execution_mask = check(execution_mask).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self._nested_obs:
            actor_features = torch.stack([self.base(obs[batch_idx]) for batch_idx in range(obs.shape[0])])
        else:
            actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        # actor_features = actor_features.reshape(bs, num_agents, -1)
        # rnn_states = rnn_states.reshape(bs, num_agents, self._recurrent_N, -1)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        
        if parents_actions is not None:
            dep_in = self.dep_fc((parents_actions * execution_mask.unsqueeze(-1)).view(*parents_actions.shape[:-2], -1))

        action_logits = self.act(actor_features, available_actions)
        if dep_mode:
            dep_final = torch.cat([actor_features, dep_in], dim=-1)
            dep_logit = self.dep_act(dep_final, available_actions)
            if self.discrete == True:
                x = action_logits.logits.detach() + dep_logit.logits
                if available_actions is not None:
                    x[available_actions == 0] = -1e10
                action_logits = FixedCategorical(logits=x)
            else:
                action_mean = action_logits.mean.detach() + dep_logit.mean
                action_std = torch.sqrt(action_logits.variance.detach() + dep_logit.variance)
                action_logits = FixedNormal(action_mean, action_std)          
            
        if deterministic:
            actions = action_logits.mode()
        else:
            actions = action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)
        # actions, action_log_probs, _ = self.act(actor_features, available_actions, deterministic, rsample=False)
        
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, parents_actions, execution_mask, dep_mode, available_actions=None, active_masks=None):
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
        if parents_actions is not None:
            parents_actions = check(parents_actions).to(**self.tpdv)
        if execution_mask is not None:
            execution_mask = check(execution_mask).to(**self.tpdv)

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

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        
        if parents_actions is not None:
            dep_in = self.dep_fc((parents_actions * execution_mask.unsqueeze(-1)).view(*parents_actions.shape[:-2], -1))

        action_logits = self.act(actor_features, available_actions)
        if dep_mode:
            dep_final = torch.cat([actor_features, dep_in], dim=-1)
            dep_logit = self.dep_act(dep_final, available_actions)
            if self.discrete == True:
                x = action_logits.logits.detach() + dep_logit.logits
                if available_actions is not None:
                    x[available_actions == 0] = -1e10
                action_logits = FixedCategorical(logits=x)
            else:
                action_mean = action_logits.mean.detach() + dep_logit.mean
                action_std = torch.sqrt(action_logits.variance.detach() + dep_logit.variance)
                action_logits = FixedNormal(action_mean, action_std)  
                
        action_log_probs = action_logits.log_probs(action)
        dist_entropy = action_logits.entropy().mean()   
        # action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions, active_masks = active_masks if self._use_policy_active_masks else None)

        values = self.v_out(actor_features) if self._use_policy_vhead else None
        
        if self._predict_other_shaped_info:
            pred_shaped_info_log_prob = self.pred_shaped_info(actor_features)
            pred_shaped_info = F.softmax(pred_shaped_info_log_prob, dim=-1)
        else:
            pred_shaped_info = None
        
        return action_log_probs, dist_entropy, values, pred_shaped_info