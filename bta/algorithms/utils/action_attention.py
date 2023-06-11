import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import init, get_clones, check
from bta.algorithms.utils.act import ACTLayer

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class Action_Attention(nn.Module):
    def __init__(self, args, action_space, device=torch.device("cpu")):
        super(Action_Attention, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._attn_N = args.attn_N
        self._gain = args.gain
        self._attn_size = args.attn_size
        self._attn_heads = args.attn_heads
        self._dropout = args.dropout
        self._use_policy_active_masks = args.use_policy_active_masks
        self.tpdv = dict(dtype=torch.float32, device=device)

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0] 

        self.logit_encoder = nn.Sequential(init_(nn.Linear(action_dim, self._attn_size), activate=True), nn.GELU())
        # self.id_encoder = nn.Sequential(init_(nn.Linear(action_dim, self._attn_size), activate=True), nn.GELU())
                
        self.mixer = MixerBlock(args.num_agents, action_dim, 
                    self._attn_size, 
                    self._dropout)
        self.mixer2 = MixerBlock(args.num_agents, action_dim, 
                    self._attn_size, 
                    self._dropout)
        self.attention = EncoderLayer(self._attn_size, self._attn_heads, self._dropout, self._use_orthogonal, self._activation_id)
            
        self.act = ACTLayer(action_space, self._attn_size, self._use_orthogonal, self._gain)
        self.to(device)

    def forward(self, x, obs_rep, mask=None, available_actions=None, deterministic=False, tau=1.0):
        bs, n_agents, action_dim = x.shape
        # logit_id = torch.eye(action_dim).unsqueeze(0).repeat(bs*n_agents, 1, 1).view(bs, n_agents*action_dim, action_dim).to(obs_rep.device)
        # id_embedding = self.id_encoder(logit_id)
        # x = self.logit_encoder(x.view(bs, n_agents*action_dim, 1)) + id_embedding
        x = self.logit_encoder(x)
        # obs_rep = obs_rep.unsqueeze(-2).repeat(1, 1, action_dim, 1).view(bs, n_agents*action_dim, -1)
        # obs_rep = obs_rep + id_embedding

        x, obs_rep = self.attention(x, obs_rep, mask)
        x = self.mixer(x)
        x = self.mixer2(x)
        
        x = x.view(bs, n_agents, -1)

        actions, action_log_probs, dist_entropy, logits = self.act(x, available_actions, deterministic, tau=tau, joint=True)
        return actions, action_log_probs
    
    def evaluate_actions(self, x, obs_rep, action, mask=None, available_actions=None, active_masks=None, tau=1.0):
        action = check(action).to(**self.tpdv)
        bs, n_agents, action_dim = x.shape
        # logit_id = torch.eye(action_dim).unsqueeze(0).repeat(bs*n_agents, 1, 1).view(bs, n_agents*action_dim, action_dim).to(obs_rep.device)
        # id_embedding = self.id_encoder(logit_id)
        # x = self.logit_encoder(x.view(bs, n_agents*action_dim, 1)) + id_embedding
        x = self.logit_encoder(x)
        # obs_rep = obs_rep.unsqueeze(-2).repeat(1, 1, action_dim, 1).view(bs, n_agents*action_dim, -1)
        # obs_rep = obs_rep + id_embedding

        x, obs_rep = self.attention(x, obs_rep, mask)
        x = self.mixer(x)
        x = self.mixer2(x)
        
        x = x.view(bs, n_agents, -1)

        train_actions, action_log_probs, _, dist_entropy, logits = self.act.evaluate_actions(x, action, available_actions, active_masks = active_masks if self._use_policy_active_masks else None, rsample=True, tau=tau, joint=True)
        
        return action_log_probs, dist_entropy, logits

class MixerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """
    def __init__(self, num_agents, action_dim, dims, 
                 dropout=0):
        super().__init__()
        self.token_layernorm = nn.LayerNorm(dims)
        self.token_forward = FeedForward(num_agents, 1, dropout)
            
        self.channel_layernorm = nn.LayerNorm(dims)
        self.channel_forward = FeedForward(dims, 4*dims, dropout)
        
    def token_mixer(self, x):
        x = self.token_layernorm(x).permute(0, 2, 1)
        x = self.token_forward(x).permute(0, 2, 1)
        return x
    
    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
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
            init_(nn.Linear(d_model, d_ff)), nn.GELU(), nn.LayerNorm(d_ff))

        self.dropout = nn.Dropout(dropout)
        self.linear_2 = init_(nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = self.dropout(self.linear_1(x))
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
        self.ff = FeedForward(d_model, 4*d_model, dropout, use_orthogonal, activation_id)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)

    def forward(self, x, obs_rep, mask=None):
        x = self.norm_1(x + self.dropout_1(self.attn1(x, x, x, mask)))
        obs_rep = self.norm_2(obs_rep + self.dropout_2(self.attn2(obs_rep, obs_rep, obs_rep, mask)))
        x = self.norm_3(obs_rep + self.dropout_3(self.attn3(k=x, v=x, q=obs_rep, mask=mask)))
        x = self.norm_4(x + self.dropout_4(self.ff(x)))

        return x, obs_rep
