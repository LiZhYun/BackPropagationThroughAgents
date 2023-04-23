import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score

################################################################################################
################################################################################################
################################################################################################

def compute_ap_score(pred_pos, pred_neg, neg_samples):
        y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu().detach()
        y_true = torch.cat([torch.ones_like(pred_pos), torch.zeros_like(pred_neg)], dim=0).cpu().detach()
        acc = average_precision_score(y_true, y_pred)
        if neg_samples > 1:
            auc = torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0)
            auc = 1 / (auc+1)
        else:
            auc = roc_auc_score(y_true, y_pred)
        return acc, auc 
    
################################################################################################
################################################################################################
################################################################################################
"""
Module: Time-encoder
"""

class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.to(torch.float32)))
        assert not torch.isinf(output).any(), "InF!!"
        return output



################################################################################################
################################################################################################
################################################################################################
"""
Module: MLP-Mixer
"""

class FeedForward(nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """
    def __init__(self, dims, expansion_factor, dropout=0, use_single_layer=False):
        super().__init__()

        self.dims = dims
        self.use_single_layer = use_single_layer
        
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        if use_single_layer:
            self.linear_0 = nn.Linear(dims, dims)
        else:
            self.linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = nn.Linear(int(expansion_factor * dims), dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        if self.use_single_layer==False:
            self.linear_1.reset_parameters()

    def forward(self, x):
        x = self.linear_0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.use_single_layer==False:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class MixerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """
    def __init__(self, per_graph_size, dims, 
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 dropout=0, 
                 module_spec=None, use_single_layer=False):
        super().__init__()
        
        if module_spec == None:
            self.module_spec = ['token', 'channel']
        else:
            self.module_spec = module_spec.split('+')

        if 'token' in self.module_spec:
            self.token_layernorm = nn.LayerNorm(dims)
            self.token_forward = FeedForward(per_graph_size, token_expansion_factor, dropout, use_single_layer)
            
        if 'channel' in self.module_spec:
            self.channel_layernorm = nn.LayerNorm(dims)
            self.channel_forward = FeedForward(dims, channel_expansion_factor, dropout, use_single_layer)
        

    def reset_parameters(self):
        if 'token' in self.module_spec:
            self.token_layernorm.reset_parameters()
            self.token_forward.reset_parameters()

        if 'channel' in self.module_spec:
            self.channel_layernorm.reset_parameters()
            self.channel_forward.reset_parameters()
        
    def token_mixer(self, x):
        x = self.token_layernorm(x).permute(0, 1, 3, 2)
        x = self.token_forward(x).permute(0, 1, 3, 2)
        return x
    
    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        if 'token' in self.module_spec:
            x = x + self.token_mixer(x)
        if 'channel' in self.module_spec:
            x = x + self.channel_mixer(x)
        return x
    
class FeatEncode(nn.Module):
    """
    Return [raw_edge_feat | TimeEncode(edge_time_stamp)]
    """
    def __init__(self, time_dims, feat_dims, out_dims):
        super().__init__()
        
        self.time_encoder = TimeEncode(time_dims)
        self.feat_encoder = nn.Linear(time_dims + feat_dims, out_dims) 
        self.reset_parameters()

    def reset_parameters(self):
        self.time_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()
        
    def forward(self, edge_feats, edge_ts, non_act_edge_mask):
        edge_time_feats = self.time_encoder(edge_ts) * non_act_edge_mask
        x = torch.cat([edge_feats, edge_time_feats], dim=-1) #[envs, agents, temporal_edges, dims]
        return self.feat_encoder(x)

class MLPMixer(nn.Module):
    """
    Input : [ batch_size, graph_size, edge_dims+time_dims]
    Output: [ batch_size, graph_size, output_dims]
    """
    def __init__(self, per_graph_size, time_channels,
                 input_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5,
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 module_spec=None, use_single_layer=False
                ):
        super().__init__()
        self.per_graph_size = per_graph_size

        self.num_layers = num_layers
        
        # input & output classifer
        self.feat_encoder = FeatEncode(time_channels, input_channels, hidden_channels)
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.mlp_head = nn.Linear(hidden_channels, out_channels)
        
        # inner layers
        self.mixer_blocks = torch.nn.ModuleList()
        for ell in range(num_layers):
            if module_spec is None:
                self.mixer_blocks.append(
                    MixerBlock(per_graph_size, hidden_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=None, 
                               use_single_layer=use_single_layer)
                )
            else:
                self.mixer_blocks.append(
                    MixerBlock(per_graph_size, hidden_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=module_spec[ell], 
                               use_single_layer=use_single_layer)
                )

        # init
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mixer_blocks:
            layer.reset_parameters()
        self.feat_encoder.reset_parameters()
        self.layernorm.reset_parameters()
        self.mlp_head.reset_parameters()

    def forward(self, edge_feats, edge_ts, non_act_edge_mask):
        # x :     [ batch_size, agents, temporal_edges, edge_dims+time_dims]
        x = self.feat_encoder(edge_feats, edge_ts, non_act_edge_mask)
        
        # apply to original feats
        for i in range(self.num_layers):
            # apply to channel + feat dim
            x = self.mixer_blocks[i](x)
        x = self.layernorm(x)
        x = torch.mean(x, dim=-2)
        x = self.mlp_head(x)
        return x
    
################################################################################################
################################################################################################
################################################################################################

"""
Edge predictor
"""

class EdgePredictor_per_node(torch.nn.Module):
    """
    out = linear(src_node_feats) + linear(dst_node_feats)
    out = ReLU(out)
    """
    def __init__(self, dim_in_time, dim_in_node, hidden_channels):
        super().__init__()

        self.dim_in_time = dim_in_time
        self.dim_in_node = dim_in_node

        self.src_fc = torch.nn.Linear(dim_in_time + dim_in_node, hidden_channels)
        self.dst_fc = torch.nn.Linear(dim_in_time + dim_in_node, hidden_channels)
        self.out_fc = torch.nn.Linear(hidden_channels, 1)
        self.reset_parameters()
        
    def reset_parameters(self, ):
        self.src_fc.reset_parameters()
        self.dst_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, h, tau=1.0): #[ batch_size, agents, edge_time_dims+node_dims]
        batch_size, num_agents = h.size(0), h.size(1)
        h_src = self.src_fc(h) # bz, num, dim
        h_dst = self.dst_fc(h)
        h_src = h_src.unsqueeze(2).repeat(1, 1, h_src.shape[1], 1) # bz, num, num, dim
        h_dst = h_dst.unsqueeze(1).repeat(1, h_dst.shape[1], 1, 1)
        h_pos_edge = torch.nn.functional.relu(h_src + h_dst)
        logits = torch.sigmoid(self.out_fc(h_pos_edge).squeeze(-1))
        adj_logits = torch.distributions.RelaxedBernoulli(torch.tensor([tau]).to(logits.device), logits=logits)
        soft_adjs = adj_logits.rsample()
        adjs = soft_adjs.round() - soft_adjs.detach() + soft_adjs
        adjs *= (1-torch.eye(num_agents).to(adjs.device)).unsqueeze(0).repeat(batch_size, 1, 1)
        
        return adjs
    
class Mixer_per_node(nn.Module):
    """
    Wrapper of MLPMixer and EdgePredictor
    """
    def __init__(self, mlp_mixer_configs, edge_predictor_configs, device=torch.device("cpu")):
        super(Mixer_per_node, self).__init__()

        self.time_feats_dim = edge_predictor_configs['dim_in_time']
        self.node_feats_dim = edge_predictor_configs['dim_in_node']

        if self.time_feats_dim > 0:
            self.base_model = MLPMixer(**mlp_mixer_configs)

        self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)
        
        self.reset_parameters()

        self.to(device)            

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()
    
    def forward(self, model_inputs, node_feats, tau=1.0):
        
        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            x = self.base_model(*model_inputs)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            x = self.base_model(*model_inputs)
            x = torch.cat([x, node_feats], dim=-1)  #[ batch_size, agents, edge_time_dims+node_dims]
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            print('Either time_feats_dim or node_feats_dim must larger than 0!')

        assert not torch.isnan(x).any(), "NaN!!"
        
        adjs = self.edge_predictor(x, tau)
        return adjs
    

################################################################################################
################################################################################################
################################################################################################

"""
Module: Node classifier
"""


class NodeClassificationModel(nn.Module):

    def __init__(self, dim_in, dim_hid, num_class):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x