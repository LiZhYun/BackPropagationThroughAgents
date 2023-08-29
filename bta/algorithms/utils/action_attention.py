import math
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import init, get_clones, check
from bta.algorithms.utils.popart import PopArt
from bta.algorithms.utils.act import ACTLayer
from bta.algorithms.utils.GPT import GPTConfig, GPT, Block
from bta.utils.util import get_shape_from_obs_space
from bta.algorithms.utils.cnn import CNNBase
from bta.algorithms.utils.mlp import MLPBase, MLPLayer
from bta.algorithms.utils.mix import MIXBase
from bta.algorithms.utils.rnn import RNNLayer
from bta.algorithms.utils.distributions import DiagGaussian

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class Action_Attention(nn.Module):
    def __init__(self, args, action_space, share_obs_space, device=torch.device("cpu")):
        super(Action_Attention, self).__init__()
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._use_popart = args.use_popart
        self._influence_layer_N = args.influence_layer_N
        self._recurrent_N = args.recurrent_N
        self._num_v_out = getattr(args, "num_v_out", 1)
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._mix_id = args.mix_id
        self._attn_N = args.attn_N
        self._gain = args.gain
        self._attn_size = args.attn_size
        self._attn_heads = args.attn_heads
        self.hidden_size = args.hidden_size
        self._dropout = args.dropout
        self._use_policy_active_masks = args.use_policy_active_masks
        self.num_agents = args.num_agents
        self.tpdv = dict(dtype=torch.float32, device=device)

        share_obs_shape = get_shape_from_obs_space(share_obs_space)
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

        self.discrete = False
        if action_space.__class__.__name__ == "Discrete":
            self.discrete = True
            action_dim = action_space.n
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0] 
            self.std_x_coef = 1.
            self.std_y_coef = 0.5
            # log_std = torch.ones(action_dim) * self.std_x_coef
            # self.log_std = torch.nn.Parameter(log_std)
        self.action_dim = action_dim

        self.logit_encoder = nn.Sequential(init_(nn.Linear(action_dim, self._attn_size), activate=True), 
                                           nn.ReLU(),
                                           nn.LayerNorm(self._attn_size))
        self.feat_encoder = nn.Sequential(init_(nn.Linear(self._attn_size+action_dim+self.num_agents, self._attn_size), activate=True), 
                                           nn.ReLU(),
                                           nn.LayerNorm(self._attn_size)
                                           )

        self.layers = nn.ModuleList()
        self.mix_type = ['mixer', 'hyper', 'attention', 'all'][self._mix_id]
        for layer in range(self._attn_N):
            if self.mix_type in 'mixer':
                self.layers.append(MixerBlock(args, args.num_agents, self._attn_heads, 
                            self._attn_size, 
                            self._dropout,
                            token_factor=args.token_factor,
                            channel_factor=args.channel_factor))
            elif self.mix_type in 'hyper':
                self.layers.append(HyperBlock(args.num_agents, action_dim, 
                            self._attn_size, 
                            self._dropout))
            elif self.mix_type in 'attention':
                self.layers.append(EncoderLayer(self._attn_size, self._attn_heads, self._dropout, 
                                                self._use_orthogonal, self._activation_id))
            else:
                config = GPTConfig(self._attn_size, self.action_dim, self.num_agents,
                        n_layer=self._attn_N, n_head=self._attn_heads, n_embd=self._attn_size)
                self.layers.append(Block(config))
                
        self.layer_norm = nn.LayerNorm(self._attn_size)
        # self.head = init_(nn.Linear(self._attn_size, self.action_dim))
        act_args = copy.copy(args)
        act_args.std_x_coef = 1
        act_args.std_y_coef = 0.2
        self.head = DiagGaussian(self._attn_size, self.action_dim, self._use_orthogonal, self._gain, act_args)

        self.to(device)

    def forward(self, x, obs_rep, rnn_states, masks):
        N = x.shape[0] // self.num_agents
        x = check(x).to(**self.tpdv)
        obs_rep = check(obs_rep).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        obs_features = self.base(obs_rep)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            obs_features, rnn_states = self.rnn(obs_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs_rep = self.mlp(obs_rep)
            obs_features = torch.cat([obs_features, mlp_obs_rep], dim=1)
        
        id_feat = torch.eye(self.num_agents).unsqueeze(0).repeat(N, 1, 1).view(-1, self.num_agents).to(x)

        x = self.feat_encoder(torch.cat([x, obs_features, id_feat], -1)).view(N, self.num_agents, -1)

        for layer in range(self._attn_N):
            x = self.layers[layer](x, obs_features.view(N, self.num_agents, -1))
        x = self.layer_norm(x)
        
        bias_ = self.head(x).rsample() 

        # action_std = None
        if self.discrete:
            action_std = torch.sigmoid(bias_)
            # action_std = -torch.exp(-bias_).log()
            action_std = -torch.log(-torch.log(action_std))
        else:
            log_std = bias_ * self.std_x_coef
            action_std = torch.sigmoid(log_std / self.std_x_coef) * self.std_y_coef

        return bias_, action_std, rnn_states.view(N, self.num_agents, self._recurrent_N, -1)
    
    def evaluation(self, x, bias, obs_rep, rnn_states, masks):
        N = x.shape[0] // self.num_agents
        x = check(x).to(**self.tpdv)
        obs_rep = check(obs_rep).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        obs_features = self.base(obs_rep)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            obs_features, rnn_states = self.rnn(obs_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs_rep = self.mlp(obs_rep)
            obs_features = torch.cat([obs_features, mlp_obs_rep], dim=1)
        
        id_feat = torch.eye(self.num_agents).unsqueeze(0).repeat(N, 1, 1).view(-1, self.num_agents).to(x)

        x = self.feat_encoder(torch.cat([x, obs_features, id_feat], -1)).view(N, self.num_agents, -1)

        for layer in range(self._attn_N):
            x = self.layers[layer](x, obs_features.view(N, self.num_agents, -1))
        x = self.layer_norm(x)
        
        bias_ = bias + self.head(x).mean.detach() - self.head(x).mean

        # action_std = None
        if self.discrete:
            action_std = torch.sigmoid(bias_)
            # action_std = -torch.exp(-bias_).log()
            action_std = -torch.log(-torch.log(action_std))
        else:
            log_std = bias_ * self.std_x_coef
            action_std = torch.sigmoid(log_std / self.std_x_coef) * self.std_y_coef

        return bias_, action_std, rnn_states.view(N, self.num_agents, self._recurrent_N, -1)

class MixerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """
    def __init__(self, args, num_agents, heads, dims, 
                 dropout=0, token_factor=0.5, channel_factor=4):
        super().__init__()
        self.h = heads
        self.dims = dims
        self.token_layernorm = nn.LayerNorm(dims)
        token_dim = int(token_factor*dims) if token_factor != 0 else 1
        self.token_forward = FeedForward(num_agents, token_dim, dropout)
            
        self.channel_layernorm = nn.LayerNorm(dims)
        channel_dim = int(channel_factor*dims) if channel_factor != 0 else 1
        self.channel_forward = FeedForward(self.dims, channel_dim, dropout)
        
    def token_mixer(self, x):
        x = self.token_layernorm(x).permute(0, 2, 1) # (10,64,2)
        x = self.token_forward(x).permute(0, 2, 1)
        return x
    
    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x, obs_rep=None):
        # if obs_rep != None:
        #     x = x + obs_rep
        x = x + self.token_mixer(x) # (10,2,64)
        x = x + self.channel_mixer(x)
        return x

class HyperBlock(nn.Module):
    def __init__(self, num_agents, action_dim, dims, 
                 dropout=0):
        super().__init__()
        self.dims = dims
        self.hyper_w1 = nn.Sequential(nn.Linear(num_agents*dims, dims),
                                          nn.ReLU(),
                                          nn.Linear(dims, num_agents*1))
        self.hyper_w12 = nn.Sequential(nn.Linear(num_agents*dims, dims),
                                          nn.ReLU(),
                                          nn.Linear(dims, num_agents*1))
        self.hyper_w2 = nn.Sequential(nn.Linear(num_agents*dims, dims),
                                        nn.ReLU(),
                                        nn.Linear(dims, dims*4*dims))
        self.hyper_w22 = nn.Sequential(nn.Linear(num_agents*dims, dims),
                                        nn.ReLU(),
                                        nn.Linear(dims, dims*4*dims))
        self.hyper_b1 = nn.Linear(num_agents*dims, 1)
        self.hyper_b12 = nn.Linear(num_agents*dims, num_agents)
        self.hyper_b2 = nn.Linear(num_agents*dims, 4*dims)
        self.hyper_b22 = nn.Linear(num_agents*dims, dims)

    def forward(self, x, obs_rep):
        bs, n_agents, action_dim = x.shape
        w1 = self.hyper_w1(obs_rep.view(bs, -1)).view(bs, n_agents, -1) # (3,2,1)
        b1 = self.hyper_b1(obs_rep.view(bs, -1)).view(bs, 1, -1)  # (3,1,1)
        hidden = F.relu(torch.bmm(x.view(bs, -1, n_agents), w1) + b1)  # (3,64,1)

        w12 = self.hyper_w12(obs_rep.view(bs, -1)).view(bs, -1, n_agents) # (3,1,2)
        b12 = self.hyper_b12(obs_rep.view(bs, -1)).view(bs, 1, -1)  # (3,1,1)
        hidden = F.relu(torch.bmm(hidden, w12) + b12).view(bs, n_agents, -1)  # (3,2,64)

        w2 = self.hyper_w2(obs_rep.view(bs, -1)).view(bs, self.dims, 4*self.dims)  # (3,64,4*64)
        b2 = self.hyper_b2(obs_rep.view(bs, -1)).view(bs, 1, 4*self.dims)  # (3,1,4*64)
        hidden = F.relu(torch.bmm(hidden, w2) + b2)  # (3, 2, 4*64)

        w22 = self.hyper_w22(obs_rep.view(bs, -1)).view(bs, self.dims*4, self.dims)  # (3,4*64,64)
        b22 = self.hyper_b22(obs_rep.view(bs, -1)).view(bs, 1, self.dims)  # (3,1,64)
        x = x + F.relu(torch.bmm(hidden, w22) + b22)  # (3, 2, 64)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.0, use_orthogonal=True, activation_id=1):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.linear_1 = nn.Sequential(
            init_(nn.Linear(d_model, d_ff)), 
            nn.ReLU(), 
            nn.LayerNorm(d_ff))
        self.dropout1 = nn.Dropout(dropout)
        self.linear_2 = init_(nn.Linear(d_ff, d_model))  

    def forward(self, x):
        x = self.dropout1(self.linear_1(x))
        x = self.linear_2(x)
        return x


def ScaledDotProductAttention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.0, use_orthogonal=True):
        super(MultiHeadAttention, self).__init__()
        
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = init_(nn.Linear(d_model, d_model))
        self.v_linear = init_(nn.Linear(d_model, d_model))
        self.k_linear = init_(nn.Linear(d_model, d_model))
        self.dropout = nn.Dropout(dropout)
        self.out = init_(nn.Linear(d_model, d_model))

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention
        scores = ScaledDotProductAttention(
            q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.0, use_orthogonal=True, activation_id=False):
        super(EncoderLayer, self).__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.attn1 = MultiHeadAttention(heads, d_model, dropout, use_orthogonal)
        self.attn2 = MultiHeadAttention(heads, d_model, dropout, use_orthogonal)
        self.ff = FeedForward(d_model, d_model, dropout, use_orthogonal, activation_id)

    def forward(self, x, obs_rep, mask=None):
        x = self.norm_1(x + self.attn1(x, x, x))
        x = self.norm_2(obs_rep + self.attn2(k=x, v=x, q=obs_rep))
        x = self.norm_3(x + self.ff(x))

        return x
