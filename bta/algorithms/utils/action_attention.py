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

        self.logit_encoder = nn.Sequential(init_(nn.Linear(action_dim, self._attn_size, bias=False), activate=True),
                                                    nn.GELU())

        self.layers = get_clones(EncoderLayer(
            self._attn_size, self._attn_heads, self._dropout, self._use_orthogonal, self._activation_id), self._attn_N)
        self.ln = nn.LayerNorm(self._attn_size)

        self.act = ACTLayer(action_space, self._attn_size, self._use_orthogonal, self._gain)
        self.to(device)

    def forward(self, x, obs_rep, mask=None, available_actions=None, deterministic=False, tau=1.0):
        x = self.logit_encoder(x) + obs_rep
        x = self.ln(x)
        for i in range(self._attn_N):
            x = self.layers[i](x, obs_rep, mask)

        actions, action_log_probs, dist_entropy, logits = self.act(x, available_actions, deterministic, tau=tau, joint=True)
        return actions, action_log_probs
    
    def evaluate_actions(self, x, obs_rep, action, mask=None, available_actions=None, active_masks=None, tau=1.0):
        action = check(action).to(**self.tpdv)
        x = self.logit_encoder(x) + obs_rep
        x = self.ln(x)
        for i in range(self._attn_N):
            x = self.layers[i](x, obs_rep, mask)

        train_actions, action_log_probs, _, dist_entropy, logits = self.act.evaluate_actions(x, action, available_actions, active_masks = active_masks if self._use_policy_active_masks else None, rsample=True, tau=tau, joint=True)
        
        return action_log_probs, dist_entropy, logits

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=128, dropout=0.0, use_orthogonal=True, activation_id=1):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.linear_1 = nn.Sequential(
            init_(nn.Linear(d_model, d_ff)), active_func, nn.LayerNorm(d_ff))

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
    def __init__(self, d_model, heads, dropout=0.0, use_orthogonal=True, activation_id=False, d_ff=128):
        super(EncoderLayer, self).__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.attn1 = MultiHeadAttention(heads, d_model, dropout, use_orthogonal)
        self.attn2 = MultiHeadAttention(heads, d_model, dropout, use_orthogonal)
        self.ff = FeedForward(d_model, d_ff, dropout, use_orthogonal, activation_id)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, obs_rep, mask):
        x = self.norm_1(x + self.dropout_1(self.attn1(x, x, x, mask)))
        # x = self.norm_2(obs_rep + self.dropout_2(self.attn2(k=x, v=x, q=obs_rep, mask=mask)))
        x = self.norm_3(x + self.dropout_3(self.ff(x)))

        return x

