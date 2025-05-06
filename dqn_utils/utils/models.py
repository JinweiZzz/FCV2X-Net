import math
import numpy as np
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLPBase
from .util import init, check
from .vit import Attention, CrossAttention, FeedForward

class EdgeGATConv(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, num_heads=2, concat=True, dropout=0.6):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.concat = concat
        
        # 每个注意力头的参数
        self.W_src = nn.Parameter(torch.Tensor(num_heads, in_dim, out_dim))
        self.W_dst = nn.Parameter(torch.Tensor(num_heads, in_dim, out_dim))
        self.W_edge = nn.Parameter(torch.Tensor(num_heads, edge_dim, out_dim))

        self.attns = nn.Parameter(torch.Tensor(num_heads, out_dim*3, 1))
        
        # 残差连接
        self.residual = nn.Linear(in_dim, num_heads * out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_src)
        nn.init.xavier_uniform_(self.W_dst)
        nn.init.xavier_uniform_(self.W_edge)
        nn.init.xavier_uniform_(self.attns)

    def forward(self, x, adj_x, adj_masks, edge_attr):
        """
        x: 中心节点特征 [1, in_dim]
        adj_x: 邻居节点特征 [n, in_dim]
        edge_attr: 边特征 [n, edge_dim]
        """
        max_neighbor = x.shape[1]
        x = x.reshape(-1, x.shape[-1]).unsqueeze(0) # [1, batch_size*max_neighbor, road_attribute_dim]
        h_src = torch.matmul(x, self.W_src)  # [num_heads, batch_size*max_neighbor, out_dim]
        adj_x = adj_x.reshape(-1, adj_x.shape[-1]).unsqueeze(0)  # [1, batch_size*max_neighbor*max_neighbor, road_attribute_dim]
        h_dst = torch.matmul(adj_x, self.W_dst)  # [num_heads, batch_size*max_neighbor*max_neighbor, out_dim]
        edge_attr = edge_attr.reshape(-1, edge_attr.shape[-1]).unsqueeze(0)  # [1, batch_size*max_neighbor*max_neighbor, edge_dim]
        h_edge = torch.matmul(edge_attr, self.W_edge)  # [num_heads, batch_size*max_neighbor*max_neighbor, out_dim]

        # h_src: [num_heads, batch_size*max_neighbor, out_dim]
        source = torch.cat([h_src.repeat(1, max_neighbor, 1), h_dst, h_edge], dim=-1)  # [num_heads, batch_size*max_neighbor*max_neighbor, 3*out_dim]
        attn_scores = torch.matmul(source, self.attns)  # [num_heads, batch_size*max_neighbor*max_neighbor, 1]
        attn_scores = attn_scores.reshape(attn_scores.shape[0], -1, max_neighbor, max_neighbor)     # [num_heads, batch_size, max_neighbor, max_neighbor]
        attn_scores = self.leaky_relu(attn_scores)
        # adj_mask: [batch_size, max_neighbor, max_neighbor]
        attn_scores[adj_masks.unsqueeze(0).repeat((self.num_heads, 1, 1, 1)) == 0] = -9e15
        attn_weights = F.softmax(attn_scores, dim=3).unsqueeze(-1)  # [num_heads, batch_size, max_neighbor, max_neighbor, 1]
        
        h_src = h_src.reshape(h_src.shape[0], -1, max_neighbor, self.out_dim)
        # h_dst*attn_weights: [num_heads, batch_size, max_neighbor, out_dim]
        h_dst = h_dst.reshape(h_dst.shape[0], -1, max_neighbor, max_neighbor, self.out_dim)
        out = h_src + (h_dst * attn_weights).sum(dim=3)  # [num_heads, batch_size, max_neighbor, out_dim]

        # 多头输出处理
        if self.concat:
            out = out.permute(1, 2, 0, 3) 
            out = torch.flatten(out, start_dim=2, end_dim=3)    # [batch_size, max_neighbor, num_heads*out_dim]
        else:
            out = out.permute(1, 2, 0, 3).mean(dim=2)  # [batch_size, max_neighbor, out_dim]
        return out
    
class EdgeConvGat(nn.Module):
    def __init__(self, neighbor_obs_dim, edge_dim):
        super(EdgeConvGat, self).__init__()
        self.edge_map = nn.Linear(edge_dim, neighbor_obs_dim)
        self.emb = nn.Linear(3*neighbor_obs_dim, neighbor_obs_dim)

    def forward(self, x, edge_attrs, adj):
        """
        x: 节点特征, 形状为 (batch_size, num_nodes, feature_dim)
        adj: 邻接矩阵, 形状为 (num_nodes, num_nodes)
        edge_attrs: 边特征, 形状为 (batch_size, num_nodes, num_nodes, edge_dim)
        """
        batch_size, num_nodes, feature_dim = x.shape
        
        x_expanded_i = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # (batch_size, num_nodes, num_nodes, feature_dim)
        x_expanded_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # (batch_size, num_nodes, num_nodes, feature_dim)
        combined_features = torch.cat([x_expanded_i, x_expanded_j], dim=-1)  # (batch_size, num_nodes, num_nodes, 2*feature_dim)      
        edge_attrs = self.edge_map(edge_attrs)  # (batch_size, num_nodes, num_nodes, neighbor_obs_dim)
        
        combined_features = torch.cat([combined_features, edge_attrs], dim=-1)  # (batch_size, num_nodes, num_nodes, feature_dim + edge_dim)
        combined_features = self.emb(combined_features)  # (batch_size, num_nodes, num_nodes, feature_dim)
        
        ### combined_features在第2、3维乘上adj
        # combined_features = combined_features * adj.unsqueeze(0).unsqueeze(-1)  # (batch_size, num_nodes, num_nodes, feature_dim + edge_dim)
        # weighted_sum = torch.sum(combined_features, dim=2)  # (batch_size, num_nodes, feature_dim + edge_dim)

        weighted_sum = torch.einsum('bijk,bij->bik', combined_features, adj)  # (batch_size, num_nodes, feature_dim)
        return weighted_sum

class EdgeConvGat_supervised(nn.Module):
    def __init__(self, neighbor_obs_dim, edge_dim):
        super(EdgeConvGat_supervised, self).__init__()
        self.edge_map = nn.Linear(edge_dim, neighbor_obs_dim)
        self.emb = nn.Linear(2*neighbor_obs_dim, neighbor_obs_dim)
        self.final_emb = nn.Linear(2*neighbor_obs_dim, neighbor_obs_dim)

    def forward(self, x, edge_attrs, adj):
        """
        x: 节点特征, 形状为 (batch_size, num_nodes, feature_dim)
        adj: 邻接矩阵, 形状为 (num_nodes, num_nodes)
        edge_attrs: 边特征, 形状为 (batch_size, num_nodes, num_nodes, edge_dim)
        """
        batch_size, num_nodes, feature_dim = x.shape
        
        x_expanded_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # (batch_size, num_nodes, num_nodes, feature_dim)  
        edge_attrs = self.edge_map(edge_attrs)  # (batch_size, num_nodes, num_nodes, neighbor_obs_dim)
        # adj = adj.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_nodes, num_nodes)
        
        combined_features = torch.cat([x_expanded_j, edge_attrs], dim=-1)  # (batch_size, num_nodes, num_nodes, feature_dim + edge_dim)
        combined_features = self.emb(combined_features)  # (batch_size, num_nodes, num_nodes, feature_dim)

        neighbor_weighted_sum = torch.einsum('bijk,bij->bik', combined_features, adj)  # (batch_size, num_nodes, feature_dim)

        combined_embs = self.final_emb(torch.cat([x, neighbor_weighted_sum], dim=-1))
        return combined_embs, neighbor_weighted_sum

class R_Actor(nn.Module):
    def __init__(self, args, source_state_dim, neighbor_state_dim, edge_dim, max_actions, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self.tpdv = dict(dtype=torch.float32, device=device)

        self._mixed_obs = False
        self._agg = args.agg
        self._corr_agg = args.corr_agg

        out_dim = 16
        num_heads = 2
        if args.agg == 1:
            self.agg = EdgeGATConv(in_dim=neighbor_state_dim, out_dim=out_dim, edge_dim=edge_dim, num_heads=2, concat=True, dropout=0)

        if args.corr_agg == 1:
            self.corr_agg = EdgeGATConv(in_dim=neighbor_state_dim, out_dim=out_dim, edge_dim=edge_dim, num_heads=2, concat=True, dropout=0)
        
        input_dim = max_actions*source_state_dim + 1        ### 加入step信息
        if self._agg == 0 and self._corr_agg == 0:
            input_dim += max_actions*neighbor_state_dim
        # self.obs_map = nn.Sequential(
        #     nn.Linear(input_dim, out_dim),
        #     nn.ReLU(),
        #     nn.Linear(out_dim, out_dim)
        # )
        self.obs_map = nn.Linear(input_dim, out_dim)

        base_dim = out_dim
        if self._agg == 1:
            base_dim += max_actions*out_dim*num_heads
        if self._corr_agg == 1:
            base_dim += max_actions*out_dim*num_heads
        self.base = MLPBase(args, base_dim, use_attn_internal=0, use_cat_self=True)
        self.action_output = nn.Linear(self.base.output_size, max_actions)

        self.to(device)

    def forward(self, obs, adj_obs, adj_neighbor_obs, adj_neighbor_masks, adj_neighbor_dest_angles_list, adj_neighbor_past_angles_list, corr_obs, corr_neighbor_obs, corr_neighbor_masks, corr_neighbor_dest_angles_list, corr_neighbor_past_angles_list):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        obs = F.relu(self.obs_map(obs))

        if self._agg == 1:
            adj_obs = check(adj_obs).to(**self.tpdv)        # batch_size * max_neighbor * road_attribute_dim
            adj_neighbor_obs = check(adj_neighbor_obs).to(**self.tpdv) # batch_size * max_neighbor * max_neighbor * road_attribute_dim
            adj_neighbor_masks = check(adj_neighbor_masks).to(**self.tpdv) # batch_size * max_neighbor * max_neighbor
            adj_neighbor_dest_angles_list = check(adj_neighbor_dest_angles_list).to(**self.tpdv).unsqueeze(-1) # batch_size * max_neighbor * max_neighbor * 1
            adj_neighbor_past_angles_list = check(adj_neighbor_past_angles_list).to(**self.tpdv).unsqueeze(-1) # batch_size * max_neighbor * max_neighbor * 1
            
            edge_features = torch.cat([adj_neighbor_past_angles_list, adj_neighbor_dest_angles_list], dim=-1) # batch_size * max_neighbor * max_neighbor * 2
            aggregated_obs = self.agg(adj_obs, adj_neighbor_obs, adj_neighbor_masks, edge_features) 
            # adj_obs = torch.cat([adj_obs, aggregated_obs], dim=-1).reshape(adj_obs.shape[0], -1) 
            obs = torch.cat([obs, aggregated_obs.reshape(aggregated_obs.shape[0], -1)], dim=-1)      ### N * max_neighbor * (num_heads * out_dim+2)

        if self._corr_agg == 1:
            corr_obs = check(corr_obs).to(**self.tpdv)        # batch_size * max_neighbor * road_attribute_dim
            corr_neighbor_obs = check(corr_neighbor_obs).to(**self.tpdv) # batch_size * max_neighbor * max_neighbor * road_attribute_dim
            corr_neighbor_masks = check(corr_neighbor_masks).to(**self.tpdv) # batch_size * max_neighbor * max_neighbor
            corr_neighbor_dest_angles_list = check(corr_neighbor_dest_angles_list).to(**self.tpdv).unsqueeze(-1) # batch_size * max_neighbor * max_neighbor * 1
            corr_neighbor_past_angles_list = check(corr_neighbor_past_angles_list).to(**self.tpdv).unsqueeze(-1) # batch_size * max_neighbor * max_neighbor * 1

            edge_features = torch.cat([corr_neighbor_past_angles_list, corr_neighbor_dest_angles_list], dim=-1) # batch_size * max_neighbor * max_neighbor * 2
            aggregated_obs = self.corr_agg(corr_obs, corr_neighbor_obs, corr_neighbor_masks, edge_features)
            obs = torch.cat([obs, aggregated_obs.reshape(aggregated_obs.shape[0], -1)], dim=-1)      ### N * max_neighbor * (num_heads * out_dim+2)
        
        obs = F.relu(self.base(obs))
        action_policy_values = self.action_output(obs)
        return action_policy_values


### Dueling DQN
class VR_Actor(nn.Module):
    def __init__(self, args, source_state_dim, neighbor_state_dim, edge_dim, max_actions, device=torch.device("cpu")):
        super(VR_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self.tpdv = dict(dtype=torch.float32, device=device)

        self._mixed_obs = False
        self._agg = args.agg
        self._corr_agg = args.corr_agg

        out_dim = 16
        num_heads = 2
        if args.agg == 1:
            self.agg = EdgeGATConv(in_dim=neighbor_state_dim, out_dim=out_dim, edge_dim=edge_dim, num_heads=2, concat=True, dropout=0)

        if args.corr_agg == 1:
            self.corr_agg = EdgeGATConv(in_dim=neighbor_state_dim, out_dim=out_dim, edge_dim=edge_dim, num_heads=2, concat=True, dropout=0)
        
        input_dim = max_actions*source_state_dim + 1        ### 加入step信息
        if self._agg == 0 and self._corr_agg == 0:
            input_dim += max_actions*neighbor_state_dim
        # self.obs_map = nn.Sequential(
        #     nn.Linear(input_dim, out_dim),
        #     nn.ReLU(),
        #     nn.Linear(out_dim, out_dim)
        # )
        self.obs_map = nn.Linear(input_dim, out_dim)

        base_dim = out_dim
        if self._agg == 1:
            base_dim += max_actions*out_dim*num_heads
        if self._corr_agg == 1:
            base_dim += max_actions*out_dim*num_heads
        self.base = MLPBase(args, base_dim, use_attn_internal=0, use_cat_self=True)
        self.action_output = nn.Linear(self.base.output_size, max_actions)
        self.value_output = nn.Linear(self.base.output_size, 1)

        self.to(device)

    def forward(self, obs, adj_obs, adj_neighbor_obs, adj_neighbor_masks, adj_neighbor_dest_angles_list, adj_neighbor_past_angles_list, corr_obs, corr_neighbor_obs, corr_neighbor_masks, corr_neighbor_dest_angles_list, corr_neighbor_past_angles_list):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        obs = F.relu(self.obs_map(obs))

        if self._agg == 1:
            adj_obs = check(adj_obs).to(**self.tpdv)        # batch_size * max_neighbor * road_attribute_dim
            adj_neighbor_obs = check(adj_neighbor_obs).to(**self.tpdv) # batch_size * max_neighbor * max_neighbor * road_attribute_dim
            adj_neighbor_masks = check(adj_neighbor_masks).to(**self.tpdv) # batch_size * max_neighbor * max_neighbor
            adj_neighbor_dest_angles_list = check(adj_neighbor_dest_angles_list).to(**self.tpdv).unsqueeze(-1) # batch_size * max_neighbor * max_neighbor * 1
            adj_neighbor_past_angles_list = check(adj_neighbor_past_angles_list).to(**self.tpdv).unsqueeze(-1) # batch_size * max_neighbor * max_neighbor * 1
            
            edge_features = torch.cat([adj_neighbor_past_angles_list, adj_neighbor_dest_angles_list], dim=-1) # batch_size * max_neighbor * max_neighbor * 2
            aggregated_obs = self.agg(adj_obs, adj_neighbor_obs, adj_neighbor_masks, edge_features) 
            # adj_obs = torch.cat([adj_obs, aggregated_obs], dim=-1).reshape(adj_obs.shape[0], -1) 
            obs = torch.cat([obs, aggregated_obs.reshape(aggregated_obs.shape[0], -1)], dim=-1)      ### N * max_neighbor * (num_heads * out_dim+2)

        if self._corr_agg == 1:
            corr_obs = check(corr_obs).to(**self.tpdv)        # batch_size * max_neighbor * road_attribute_dim
            corr_neighbor_obs = check(corr_neighbor_obs).to(**self.tpdv) # batch_size * max_neighbor * max_neighbor * road_attribute_dim
            corr_neighbor_masks = check(corr_neighbor_masks).to(**self.tpdv) # batch_size * max_neighbor * max_neighbor
            corr_neighbor_dest_angles_list = check(corr_neighbor_dest_angles_list).to(**self.tpdv).unsqueeze(-1) # batch_size * max_neighbor * max_neighbor * 1
            corr_neighbor_past_angles_list = check(corr_neighbor_past_angles_list).to(**self.tpdv).unsqueeze(-1) # batch_size * max_neighbor * max_neighbor * 1

            edge_features = torch.cat([corr_neighbor_past_angles_list, corr_neighbor_dest_angles_list], dim=-1) # batch_size * max_neighbor * max_neighbor * 2
            aggregated_obs = self.corr_agg(corr_obs, corr_neighbor_obs, corr_neighbor_masks, edge_features)
            obs = torch.cat([obs, aggregated_obs.reshape(aggregated_obs.shape[0], -1)], dim=-1)      ### N * max_neighbor * (num_heads * out_dim+2)
        
        obs = F.relu(self.base(obs))
        A = self.action_output(obs)
        V = self.value_output(obs)
        action_policy_values = V + A - A.mean(dim=1, keepdim=True)     # Dueling DQN
        return action_policy_values
    
class Invariant(nn.Module):
    def __init__(self, input_dim, invariant_type = "type_sigmoid_attn", hidden_dim=128, heads=4, dim_head=32, mlp_dim=128, dropout=0., depth=1, distance=0, neighbor_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.invariant_type = invariant_type
        self.depth = depth
        self.distance = distance
        if invariant_type == 'type_sigmoid_attn':
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(neighbor_dim, hidden_dim)
            self.spatial_attn_layer_x = nn.ModuleList([
                Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            self.type_attn_layer = nn.ModuleList([
                CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout, embedding_dimension_query=2),     ### 用type做query去提取信息
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            self.spatial_attn_layer_y = nn.ModuleList([
                Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            if self.distance == 0:
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])
            elif self.distance == 1:
                self.encode_distance_net = nn.Linear(2, hidden_dim)
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])
            elif self.distance == 2:
                self.encode_distance_net = nn.Linear(2, 8)
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout, embedding_dimension_query=8+hidden_dim),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])               
            self.map_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            self.fc_sum = nn.Linear(2*hidden_dim, hidden_dim)
        else:
            raise NotImplementedError
    
    def encode_actor(self, x):
        fc = self.encode_actor_net
        return fc(x)
    
    def encode_other(self, y):
        fc = self.encode_other_net
        return fc(y)

    def attn_type(self, x):
        x, x_type = x
        attn, ff = self.type_attn_layer
        x = attn(x, x_type) + x     
        x = ff(x)       ## feed forward layer
        return x

    def attn_cross_distance(self, x):
        x, main = x
        attn, ff = self.cross_attn_layer
        x = attn(x, main) + x
        # x = attn(x, main)
        # x = ff(x) + x 
        x = ff(x)
        return x
    
    def attn_self_x(self, x):
        attn, ff = self.spatial_attn_layer_x
        # x = attn(x)
        x = attn(x) + x
        x = ff(x)
        return x
    
    def attn_self_y(self, x):
        attn, ff = self.spatial_attn_layer_y
        x = attn(x) + x
        # x = attn(x)
        x = ff(x) 
        return x

    def forward(self, x, others, neighbor_masks=None, neighbor_types=None, neighbor_relations=None, neighbor_distances=None):
        B = x.shape[0]
        assert len(others) == len(neighbor_masks) == len(neighbor_types) == len(neighbor_relations) == len(neighbor_distances) == 4
        if self.invariant_type == 'type_sigmoid_attn':
            x = self.encode_actor(x)
            x = self.attn_self_x(x)
            out_typea, out_typeb = torch.zeros(B, 2, self.hidden_dim).to(x.device), torch.zeros(B, 2, self.hidden_dim).to(x.device)
            coef_typea, coef_typeb = torch.zeros(B, 2, 1).to(x.device), torch.zeros(B, 2, 1).to(x.device)
            for i in range(2):
                y, neighbor_mask, neighbor_relation, neighbor_distance = others[i], neighbor_masks[i], neighbor_relations[i], neighbor_distances[i]
                y = self.encode_other(y)
                y = self.attn_type((y, neighbor_relation))
                y = self.attn_self_y(y)
                if self.distance == 1:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = x + neighbor_distance
                    z = self.attn_cross_distance((y, x_distance))
                elif self.distance == 2:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = torch.cat([x, neighbor_distance], dim=1)
                    z = self.attn_cross_distance((y, x_distance))
                else:
                    z = self.attn_cross_distance((y, x))
                z_sig = self.map_layer(z)
                z_sig[neighbor_mask==0, :] = -np.inf
                out_typea[:, i, :] = z
                coef_typea[:, i, :] = z_sig
            coef_typea = torch.softmax(coef_typea, dim=1)
            coef_typea = torch.nan_to_num(coef_typea, nan=0.0)
            out_typea = torch.sum(out_typea * coef_typea, dim=1)
            for i in range(2, 4):
                y, neighbor_mask, neighbor_relation, neighbor_distance = others[i], neighbor_masks[i], neighbor_relations[i], neighbor_distances[i]
                y = self.encode_other(y)
                y = self.attn_type((y, neighbor_relation))
                y = self.attn_self_y(y)
                if self.distance == 1:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = x + neighbor_distance
                    z = self.attn_cross_distance((y, x_distance))
                elif self.distance == 2:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = torch.cat([x, neighbor_distance], dim=1)
                    z = self.attn_cross_distance((y, x_distance))
                else:
                    z = self.attn_cross_distance((y, x))
                z_sig = self.map_layer(z)
                z_sig[neighbor_mask==0, :] = -np.inf
                out_typeb[:, i-2, :] = z
                coef_typeb[:, i-2, :] = z_sig
            coef_typeb = torch.softmax(coef_typeb, dim=1)
            coef_typeb = torch.nan_to_num(coef_typeb, nan=0.0)
            out_typeb = torch.sum(out_typeb * coef_typeb, dim=1)
            return x, self.fc_sum(torch.cat([out_typea, out_typeb], dim=-1))
        else:
            raise NotImplementedError


class J_Actor(nn.Module):
    def __init__(self, args, obs_dim, action_dim, device=torch.device("cpu")):
        super(J_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self.attn = args.attn
        self.distance = args.distance
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = obs_dim
        self.type_idx_a = [0, 1, 2, 3, 4, 5, 6, 7]
        self.type_idx_b = [2, 3, 0, 1, 6, 7, 4, 5]
        
        self._mixed_obs = False
        hidden_dim = 64
        dim_head = 32
        mlp_dim = 32
        self.neighbor_dim = len(self.type_idx_a)
        
        if args.junction_agg == 1:
            self.agg = Invariant(obs_shape, invariant_type=args.attn, hidden_dim=hidden_dim, dim_head=dim_head, mlp_dim=mlp_dim, distance=args.distance, neighbor_dim=self.neighbor_dim)
        self._agg = 1
        if self._agg:
            obs_shape = hidden_dim * 2
        print('obs Shape: ', obs_shape)
        self.base = MLPBase(args, obs_shape, use_attn_internal=args.use_attn_internal, use_cat_self=True)
        self.action_output = nn.Linear(self.base.output_size, action_dim)

        self.to(device)

    def forward(self, obs, neighbor_obs, neighbor_mask, neighbor_type, neighbor_relation, neighbor_distance):
        obs = check(obs).to(**self.tpdv)

        if self._agg:
            other_obs = [check(neighbor_obs[:, i]).to(**self.tpdv)[:, self.type_idx_a] for i in range(neighbor_obs.shape[-2])]
            masks = [neighbor_mask[:, i] for i in range(neighbor_mask.shape[-1])]
            other_types = [neighbor_type[:, i] for i in range(neighbor_type.shape[-1])]
            other_relations = [check(neighbor_relation[:, i]).to(**self.tpdv) for i in range(neighbor_relation.shape[-2])]
            other_distances = [check(neighbor_distance[:, i]).to(**self.tpdv) for i in range(neighbor_distance.shape[-2])]   
            obs, obs_n = self.agg(obs, other_obs, masks, other_types, other_relations, other_distances)
            obs = torch.cat([obs, obs_n], dim=-1)       

        actor_features = F.relu(self.base(obs))
        action_policy_values = self.action_output(actor_features)

        return action_policy_values
    

class VJ_Actor(nn.Module):
    def __init__(self, args, obs_dim, action_dim, device=torch.device("cpu")):
        super(VJ_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self.attn = args.attn
        self.distance = args.distance
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = obs_dim
        self.type_idx_a = [0, 1, 2, 3, 4, 5, 6, 7]
        self.type_idx_b = [2, 3, 0, 1, 6, 7, 4, 5]
        
        self._mixed_obs = False
        hidden_dim = 64
        dim_head = 32
        mlp_dim = 32
        self.neighbor_dim = len(self.type_idx_a)
        
        if args.junction_agg == 1:
            self.agg = Invariant(obs_shape, invariant_type=args.attn, hidden_dim=hidden_dim, dim_head=dim_head, mlp_dim=mlp_dim, distance=args.distance, neighbor_dim=self.neighbor_dim)
        self._agg = 1
        if self._agg:
            obs_shape = hidden_dim * 2
        print('obs Shape: ', obs_shape)
        self.base = MLPBase(args, obs_shape, use_attn_internal=args.use_attn_internal, use_cat_self=True)
        self.action_output = nn.Linear(self.base.output_size, action_dim)
        self.value_output = nn.Linear(self.base.output_size, 1)

        self.to(device)

    def forward(self, obs, neighbor_obs, neighbor_mask, neighbor_type, neighbor_relation, neighbor_distance):
        obs = check(obs).to(**self.tpdv)

        if self._agg:
            other_obs = [check(neighbor_obs[:, i]).to(**self.tpdv)[:, self.type_idx_a] for i in range(neighbor_obs.shape[-2])]
            masks = [neighbor_mask[:, i] for i in range(neighbor_mask.shape[-1])]
            other_types = [neighbor_type[:, i] for i in range(neighbor_type.shape[-1])]
            other_relations = [check(neighbor_relation[:, i]).to(**self.tpdv) for i in range(neighbor_relation.shape[-2])]
            other_distances = [check(neighbor_distance[:, i]).to(**self.tpdv) for i in range(neighbor_distance.shape[-2])]   
            obs, obs_n = self.agg(obs, other_obs, masks, other_types, other_relations, other_distances)
            obs = torch.cat([obs, obs_n], dim=-1)       

        actor_features = F.relu(self.base(obs))
        A = self.action_output(actor_features)
        V = self.value_output(actor_features)
        action_policy_values = V + A - A.mean(dim=1, keepdim=True)     # Dueling DQN

        return action_policy_values
    
class BGCN_Actor(nn.Module):
    def __init__(self, args, source_obs_dim, obs_dim, edge_dim, max_action, roadidx2neighboridxs, device=torch.device("cpu")):
        super(BGCN_Actor, self).__init__()
        self.num_roads = args.num_roads
        self.roadidx2adjidxs = roadidx2neighboridxs
        self.agg_type = args.agg_type
        self.supervised = args.supervised_signal
        A = np.zeros((self.num_roads, self.num_roads))
        for i in range(self.num_roads):
            for j in range(self.num_roads):
                if i in self.roadidx2adjidxs[j]:
                    A[i][j] = 1
        self.road_neighbor_idxs = np.zeros((self.num_roads, max_action))
        self.road_neighbor_masks = np.zeros((self.num_roads, max_action))
        for i in range(self.num_roads):
            for jidx, j in enumerate(self.roadidx2adjidxs[i]):
                self.road_neighbor_idxs[i][jidx] = j
                self.road_neighbor_masks[i][jidx] = 1
        self.road_neighbor_idxs = torch.from_numpy(self.road_neighbor_idxs.astype(np.int64)).to(device)
        self.road_neighbor_masks = torch.from_numpy(self.road_neighbor_masks.astype(np.int64)).to(device)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)).to(device), requires_grad=False)
        self.d = 0.2
        self.mean_field = args.mean_field

        out_dim = 16
        self.source_obs_map = nn.Linear(source_obs_dim, out_dim*max_action)
        if args.mean_field:
            self.obs_map = nn.Linear(obs_dim, out_dim)
        else:
            self.obs_map = nn.Linear(obs_dim-1, out_dim)

        if args.agg_type == 'bgcn':
            self.PA = nn.Parameter(torch.ones(self.num_roads, self.num_roads))   # global 
            nn.init.constant_(self.PA, 1e-6)

        self.conv_times = 2
        self.gconv = nn.ModuleList()
        self.fixed_adj_gconv = nn.ModuleList()
        self.residual = nn.ModuleList()
        self.fixed_adj_residual = nn.ModuleList()
        for i in range(self.conv_times):
            self.residual.append(nn.Linear(out_dim, out_dim))
            if args.supervised_signal == 0:
                self.gconv.append(EdgeConvGat(out_dim, edge_dim))
            elif args.supervised_signal == 1:
                self.gconv.append(EdgeConvGat_supervised(out_dim, edge_dim))
            self.fixed_adj_residual.append(nn.Linear(out_dim, out_dim))
            self.fixed_adj_gconv.append(EdgeConvGat(out_dim, edge_dim))

        base_dim = out_dim * max_action * 4
        self.base = MLPBase(args, base_dim, use_attn_internal=0, use_cat_self=True)
        self.action_output = nn.Linear(self.base.output_size, max_action)

        self.supervised_prediction_layer = nn.Linear(out_dim, out_dim)

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.to(device)

    def forward(self, obs, obs_all, edge_attrs, ridxs, corr_adj_matrix, training=True):
        if self.agg_type == 'bgcn':
            adaptive_adj = F.dropout(self.A+self.PA, self.d, training=training)
            adaptive_adj = adaptive_adj.unsqueeze(0).expand(obs.shape[0], -1, -1)  # (batch_size, num_nodes, num_nodes)

        obs = check(obs).to(**self.tpdv)        ### sample_size, obs_dim
        obs = self.source_obs_map(obs)        ### sample_size, out_dim

        obs_all = check(obs_all).to(**self.tpdv)        ### sample_size, num_roads, obs_dim
        edge_attrs = check(edge_attrs).to(**self.tpdv)        ### sample_size, num_roads, num_roads, 1
        if self.mean_field == 0:
            obs_all = obs_all[:, :, :-1]        ### sample_size, num_roads, obs_dim - 1
        obs_all = self.obs_map(obs_all)        ### sample_size, num_roads, out_dim
        row_indices_expanded = self.road_neighbor_idxs[ridxs].unsqueeze(2).expand(-1, -1, obs_all.shape[-1])  # (sample_size, max_action, out_dim)
        obs_all_origin_selected = torch.gather(obs_all, 1, row_indices_expanded)   # (sample_size, max_action, out_dim)

        for i in range(self.conv_times):
            obs_all_residual = self.residual[i](obs_all)
            if self.agg_type == 'bgcn':
                if self.supervised == 0:
                    aggregated_rep = self.gconv[i](obs_all, edge_attrs, adaptive_adj)
                    obs_all_adaptive = F.relu(obs_all_residual + aggregated_rep)
                else:
                    aggregated_rep, neighbor_rep = self.gconv[i](obs_all, edge_attrs, adaptive_adj)
                    obs_all_adaptive = F.relu(obs_all_residual + aggregated_rep)
            elif self.agg_type == 'corr_agg':
                obs_all_adaptive = F.relu(obs_all_residual + self.gconv[i](obs_all, edge_attrs, corr_adj_matrix))
            else:
                obs_all_adaptive = None 
            obs_all_residual_fixed = self.fixed_adj_residual[i](obs_all)
            obs_all_fixed = F.relu(obs_all_residual_fixed + self.fixed_adj_gconv[i](obs_all, edge_attrs, self.A.unsqueeze(0).expand(obs.shape[0], -1, -1))) 

        if self.supervised == 1:
            neighbor_rep = self.supervised_prediction_layer(neighbor_rep)
            neighbor_rep = torch.gather(neighbor_rep, 1, row_indices_expanded)  # (sample_size, max_action, out_dim)
            neighbor_rep[self.road_neighbor_masks[ridxs] == 0] = -1
            neighbor_rep = neighbor_rep.reshape(neighbor_rep.shape[0], -1)  # (sample_size, max_action * out_dim)
        obs_all_selected = torch.concatenate((torch.gather(obs_all_adaptive, 1, row_indices_expanded), torch.gather(obs_all_fixed, 1, row_indices_expanded)), dim=-1)  # (sample_size, max_action, out_dim * 2)
        obs_all_selected[self.road_neighbor_masks[ridxs] == 0] = -1
        obs_all_selected = obs_all_selected.reshape(obs_all_selected.shape[0], -1)  # (sample_size, max_action * out_dim * 2)
        obs_all_origin_selected[self.road_neighbor_masks[ridxs] == 0] = -1
        obs_all_origin_selected = obs_all_origin_selected.reshape(obs_all_origin_selected.shape[0], -1)  # (sample_size, max_action * out_dim)

        obs = torch.concatenate((obs, obs_all_origin_selected, obs_all_selected), dim=-1)  # (batch_size, max_action * out_dim * 4)
        obs = F.relu(self.base(obs))
        action_policy_values = self.action_output(obs)
        
        if self.supervised == 1:
            return action_policy_values, neighbor_rep, obs_all_origin_selected
        else:
            return action_policy_values
                