import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import init, get_clones, check
from bta.algorithms.utils.popart import PopArt
from bta.algorithms.utils.act import ACTLayer

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class Action_Attention(nn.Module):
    def __init__(self, args, action_space, device=torch.device("cpu")):
        super(Action_Attention, self).__init__()
        self._use_popart = args.use_popart
        self._num_v_out = getattr(args, "num_v_out", 1)
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._mix_id = args.mix_id
        self._attn_N = args.attn_N
        self._gain = args.gain
        self._attn_size = args.attn_size
        self._attn_heads = args.attn_heads
        self._dropout = args.dropout
        self._use_policy_active_masks = args.use_policy_active_masks
        self.num_agents = args.num_agents
        self.tpdv = dict(dtype=torch.float32, device=device)

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0] 
        self.action_dim = action_dim

        self.logit_encoder = nn.Sequential(init_(nn.Linear(action_dim, self._attn_size), activate=True), 
                                           nn.ReLU(),
                                           nn.LayerNorm(self._attn_size))
        self.id_encoder = nn.Sequential(init_(nn.Linear(self.num_agents, self._attn_size), activate=True), 
                                           nn.ReLU(),
                                           nn.LayerNorm(self._attn_size))
        self.feat_encoder = nn.Sequential(init_(nn.Linear(self._attn_size+self.action_dim, self._attn_size), activate=True), 
                                           nn.ReLU(),
                                           nn.LayerNorm(self._attn_size),
                                        #    init_(nn.Linear(self._attn_size, self._attn_size), activate=True), 
                                        #    nn.ReLU(),
                                        #    nn.LayerNorm(self._attn_size),
                                           )
        
        self.layers = nn.ModuleList()
        mix_type = ['mixer', 'hyper', 'attention', 'all'][self._mix_id]
        for layer in range(self._attn_N):
            if mix_type in 'mixer':
                self.layers.append(MixerBlock(args, args.num_agents, self._attn_heads, 
                            self._attn_size, 
                            self._dropout))
            elif mix_type in 'hyper':
                self.layers.append(HyperBlock(args.num_agents, action_dim, 
                            self._attn_size, 
                            self._dropout))
            elif mix_type in 'attention':
                self.layers.append(EncoderLayer(self._attn_size, self._attn_heads, self._dropout, 
                                                self._use_orthogonal, self._activation_id))
            else:
                self.layers.extend([MixerBlock(args.num_agents, action_dim, 
                            self._attn_size, 
                            self._dropout),

                            HyperBlock(args.num_agents, action_dim, 
                            self._attn_size, 
                            self._dropout),

                            EncoderLayer(self._attn_size, self._attn_heads, 
                                         self._dropout, self._use_orthogonal, 
                                         self._activation_id)]
                            )
        # self.act = ACTLayer(action_space, self._attn_size, self._use_orthogonal, self._gain)
        self.layer_norm = nn.LayerNorm(self._attn_size)
        self.linear = nn.Sequential(init_(nn.Linear(self._attn_size+self.num_agents, self._attn_size), activate=True), 
                                           nn.ReLU(),
                                           nn.LayerNorm(self._attn_size),
                                           init_(nn.Linear(self._attn_size, action_dim), activate=True), 
                                           )
        # if self._use_popart:
        #     self.v_out = init_(PopArt(self._attn_size, self._num_v_out, device=device))
        # else:
        #     self.v_out = init_(nn.Linear(self._attn_size, self._num_v_out))
        self.to(device)

    def forward(self, x, obs_rep, mask=None, available_actions=None, deterministic=False, tau=1.0):
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        x = self.feat_encoder(torch.cat([x, obs_rep], -1))

        xs_ = []
        for agent_id in range(self.num_agents):
            x_ = torch.cat([x[:, agent_id].unsqueeze(1), torch.cat([x[:, :agent_id], x[:, agent_id+1:]],1)],1)
            xs_.append(x_)

        x = torch.stack(xs_, 1)

        for layer in range(self._attn_N):
            x = self.layers[layer](x, obs_rep)
        x = self.layer_norm(x)
        x = torch.mean(x, dim=2)
        
        id_feat = torch.eye(self.num_agents).unsqueeze(0).repeat(x.shape[0], 1, 1).to(x.device)
        x = torch.cat([x, id_feat], -1)

        bias_ = self.linear(x)
        # values = self.v_out(x)

        # actions, action_log_probs, dist_entropy, logits = self.act(x, available_actions, deterministic, tau=tau, joint=True)
        return bias_

class MixerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """
    def __init__(self, args, num_agents, heads, dims, 
                 dropout=0):
        super().__init__()
        self.h = heads
        self.dims = dims
        self.token_layernorm = nn.LayerNorm(dims)
        token_dim = int(args.token_factor*dims) if args.token_factor != 0 else 1
        self.token_forward = FeedForward(num_agents, token_dim, dropout)
        # self.token_forward = nn.ModuleList()
        # for _ in range(self.h):
        #     self.token_forward.append(FeedForward(num_agents, token_dim, dropout))
            
        self.channel_layernorm = nn.LayerNorm(dims)
        self.channel_forward = FeedForward(self.dims, 4*self.dims, dropout)
        # self.channel_forward = nn.ModuleList()
        # for _ in range(self.h):
        #     self.channel_forward.append(FeedForward(self.dims, 4*self.dims, dropout))
        
    def token_mixer(self, x):
        x = self.token_layernorm(x).permute(0, 1, 3, 2) # (10,64,2)
        x = self.token_forward(x).permute(0, 1, 3, 2)
        # out = torch.zeros_like(x).permute(0, 2, 1).to(x)
        # for head in range(self.h):
            # out += self.token_forward[head](x).permute(0, 2, 1)
        return x
    
    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        # out = torch.zeros_like(x).to(x)
        # for head in range(self.h):
        #     out += self.channel_forward[head](x)
        return x

    def forward(self, x, obs_rep):
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
        hidden = F.ReLU(torch.bmm(x.view(bs, -1, n_agents), w1) + b1)  # (3,64,1)

        w12 = self.hyper_w12(obs_rep.view(bs, -1)).view(bs, -1, n_agents) # (3,1,2)
        b12 = self.hyper_b12(obs_rep.view(bs, -1)).view(bs, 1, -1)  # (3,1,1)
        hidden = F.ReLU(torch.bmm(hidden, w12) + b12).view(bs, n_agents, -1)  # (3,2,64)

        w2 = self.hyper_w2(obs_rep.view(bs, -1)).view(bs, self.dims, 4*self.dims)  # (3,64,4*64)
        b2 = self.hyper_b2(obs_rep.view(bs, -1)).view(bs, 1, 4*self.dims)  # (3,1,4*64)
        hidden = F.ReLU(torch.bmm(hidden, w2) + b2)  # (3, 2, 4*64)

        w22 = self.hyper_w22(obs_rep.view(bs, -1)).view(bs, self.dims*4, self.dims)  # (3,4*64,64)
        b22 = self.hyper_b22(obs_rep.view(bs, -1)).view(bs, 1, self.dims)  # (3,1,64)
        x = x + F.ReLU(torch.bmm(hidden, w22) + b22)  # (3, 2, 64)
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
        self.norm_4 = nn.LayerNorm(d_model)
        self.attn1 = MultiHeadAttention(heads, d_model, dropout, use_orthogonal)
        self.attn2 = MultiHeadAttention(heads, d_model, dropout, use_orthogonal)
        self.attn3 = MultiHeadAttention(heads, d_model, dropout, use_orthogonal)
        self.ff = FeedForward(d_model, d_model, dropout, use_orthogonal, activation_id)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)

    def forward(self, x, obs_rep, mask=None):
        x = self.norm_1(x + self.dropout_1(self.attn1(x, x, x, mask)))
        x = self.norm_4(x + self.dropout_4(self.ff(x)))

        return x
