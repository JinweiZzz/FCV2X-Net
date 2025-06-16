import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
from torch.distributions import Categorical
from torch.autograd import Variable

from .mlp import MLPBase
from .util import init, check
from torch.nn.init import xavier_normal_

class FeedforwardMLP(nn.Module):
    def __init__(self, out_dim: int, hidden_dim: int, output_activation: Optional[str] = None):
        super().__init__()

        self._hidden_layer = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self._output_layer = nn.Linear(in_features=hidden_dim, out_features=out_dim)
        self._output_activation = output_activation

    def forward(self, inputs: Tensor) -> Tensor:
        x = F.relu(self._hidden_layer(inputs))
        if self._output_activation:
            return F.relu(self._output_layer(x))
        return self._output_layer(x)

class J_Actor(nn.Module):
    def __init__(self, args, obs_dim, action_dim, device=torch.device("cpu")):
        super(J_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self.attn = args.attn
        self.distance = args.distance
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = obs_dim
        
        self._mixed_obs = False
        
        self.base = MLPBase(args, obs_shape, use_attn_internal=args.use_attn_internal, use_cat_self=True)
        self.action_output = nn.Linear(self.base.output_size, action_dim)

        self.to(device)

    def forward(self, obs, neighbor_obs, neighbor_mask, neighbor_type, neighbor_relation, neighbor_distance):
        obs = check(obs).to(**self.tpdv)   

        actor_features = F.relu(self.base(obs))
        action_policy_values = self.action_output(actor_features)

        return action_policy_values

class MultiHeadAttention(nn.Module):
    def __init__(self, output_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.output_dim = output_dim

        self.query = nn.Linear(output_dim, output_dim)
        self.key = nn.Linear(output_dim, output_dim)
        self.value = nn.Linear(output_dim, output_dim)
        self.attn_output = nn.Linear(output_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Shape of x is (batch_size, seq_len, output_dim)
        batch_size, seq_len, _ = x.size()

        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, value)
        output = output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        return self.attn_output(output), attention_weights

class XRoutingModel(nn.Module):
    def __init__(self,
                 observation_dim: int,
                 pos_encoding_dim: int,
                 num_outputs: int,
                 attention_dim: int = 64,
                 num_heads: int = 2,
                 head_dim: int = 32,
                 mlp_dim: int = 32):
        super().__init__()

        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.obs_dim = observation_dim
        self.pos_encoding_dim = pos_encoding_dim
        self.num_outputs = num_outputs

        # Observation input layer
        self.input_layer = nn.Linear(self.obs_dim, self.attention_dim//2)

        # Position encoding input layer
        self.pos_encoding_input = nn.Linear(self.pos_encoding_dim, self.attention_dim // 2)

        # Multi-Head Attention
        self.MHA = MultiHeadAttention(self.attention_dim, self.num_heads, self.head_dim)

        # GRU layers (using nn.GRU for simplicity)
        self.GRU_1 = nn.GRU(input_size=self.attention_dim, hidden_size=self.attention_dim, batch_first=True)
        self.GRU_2 = nn.GRU(input_size=self.attention_dim, hidden_size=self.attention_dim, batch_first=True)

        # Layer normalization
        self.layernorm = nn.LayerNorm(self.attention_dim)

        # Feedforward MLP
        self.mlp = FeedforwardMLP(out_dim=self.attention_dim, hidden_dim=self.attention_dim)

        # Final linear layer and value head
        self.final_linear = nn.Linear(self.attention_dim*self.num_outputs, 64)
        self.logits = nn.Linear(64, self.num_outputs)
        # self.values = nn.Linear(64, 1)

    def forward(self, observations: Tensor, position: Tensor, available_actions: Optional[Tensor] = None) -> Tensor:
        observation_embedding = self.input_layer(observations)
        position_embedding = self.pos_encoding_input(position.unsqueeze(-1))
        E_out = torch.cat([observation_embedding, position_embedding], dim=-1)

        attention_out, _ = self.MHA(E_out)
        gru_out_1, _ = self.GRU_1(attention_out)
        norm_out_1 = self.layernorm(gru_out_1)
        mlp_out = self.mlp(norm_out_1)
        gru_out_2, _ = self.GRU_2(mlp_out)
        flatten_out = gru_out_2.flatten(start_dim=1)

        out = F.relu(self.final_linear(flatten_out))
        logits = self.logits(out)

        if available_actions is not None:
            neg_inf = torch.finfo(logits.dtype).min
            logits = torch.where(available_actions.bool(), logits, neg_inf)

        dist = Categorical(logits=logits)
        values = self.values(out)
        return dist, values

    def act(self, obs, available_actions=None):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        mask_tensor = None
        if available_actions is not None:
            mask_tensor = torch.as_tensor(available_actions, dtype=torch.float32)
        dist, value  = self.forward(obs_tensor, mask_tensor)
        action       = dist.sample()
        log_prob     = dist.log_prob(action)
        return action.cpu().numpy(), log_prob, value

    def evaluate_actions(self, obs, actions, available_actions=None):
        dist, value = self.forward(obs, available_actions)
        log_prob    = dist.log_prob(actions)
        entropy     = dist.entropy()
        return log_prob, entropy, value
    
class XRoutingModel_DQN(nn.Module):
    def __init__(self,
                 observation_dim: int,
                 pos_encoding_dim: int,
                 num_outputs: int,
                 attention_dim: int = 64,
                 num_heads: int = 2,
                 head_dim: int = 32,
                 mlp_dim: int = 32):
        super().__init__()

        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.obs_dim = observation_dim
        self.pos_encoding_dim = pos_encoding_dim
        self.num_outputs = num_outputs

        # Observation input layer
        self.input_layer = nn.Linear(self.obs_dim, self.attention_dim//2)

        # Position encoding input layer
        self.pos_encoding_input = nn.Linear(self.pos_encoding_dim, self.attention_dim // 2)

        # Multi-Head Attention
        self.MHA = MultiHeadAttention(self.attention_dim, self.num_heads, self.head_dim)

        # GRU layers (using nn.GRU for simplicity)
        self.GRU_1 = nn.GRU(input_size=self.attention_dim, hidden_size=self.attention_dim, batch_first=True)
        # self.GRU_2 = nn.GRU(input_size=self.attention_dim, hidden_size=self.attention_dim, batch_first=True)

        # Layer normalization
        self.layernorm = nn.LayerNorm(self.attention_dim)

        # Feedforward MLP
        self.mlp = FeedforwardMLP(out_dim=self.attention_dim, hidden_dim=self.attention_dim)

        # Final linear layer and value head
        self.final_linear = nn.Linear(self.attention_dim*self.num_outputs, 64)
        # self.final_linear = nn.Linear(self.attention_dim, 64)
        self.logits = nn.Linear(64, self.num_outputs)
        # self.values = nn.Linear(64, 1)

    def forward(self, observations: Tensor, position: Tensor, available_actions: Optional[Tensor] = None) -> Tensor:
        observation_embedding = self.input_layer(observations)
        position_embedding = self.pos_encoding_input(position.unsqueeze(-1))
        E_out = torch.cat([observation_embedding, position_embedding], dim=-1)

        attention_out, _ = self.MHA(E_out)
        gru_out_1, _ = self.GRU_1(attention_out)
        flatten_out = gru_out_1.flatten(start_dim=1)
        # norm_out_1 = self.layernorm(gru_out_1)
        # mlp_out = self.mlp(norm_out_1)
        # mlp_out = mlp_out.flatten(start_dim=1)
        # gru_out_2, _ = self.GRU_2(mlp_out)
        # flatten_out = gru_out_2.flatten(start_dim=1)

        out = F.relu(self.final_linear(flatten_out))
        logits = self.logits(out)

        return logits

class ANModel(nn.Module):
    def __init__(self, args, source_obs_dim, obs_dim, edge_dim, max_action, roadidx2neighboridxs, device=torch.device("cpu")):
        super(ANModel, self).__init__()
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
        self.residual = nn.ModuleList()
        for i in range(self.conv_times):
            self.residual.append(nn.Linear(out_dim, out_dim))
            if args.supervised_signal == 0:
                self.gconv.append(EdgeConvGat(out_dim, edge_dim))
            elif args.supervised_signal == 1:
                self.gconv.append(EdgeConvGat_supervised(out_dim, edge_dim))

        if args.agg_type == 'bgcn':
            base_dim = out_dim * max_action * 2
            self.gated_layer = nn.Linear(base_dim, out_dim*2)
        elif args.agg_type == 'none':
            base_dim = out_dim * max_action
            self.gated_layer = nn.Linear(base_dim, out_dim*2)
        else:
            raise NotImplementedError

        self.base = MLPBase(args, out_dim, use_attn_internal=0, use_cat_self=True)
        self.action_output = nn.Linear(self.base.output_size, max_action)

        self.supervised_prediction_layer = nn.Linear(out_dim, out_dim)

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.to(device)

    def forward(self, obs, obs_all, edge_attrs, ridxs, corr_adj_matrix, training=True):
        if self.agg_type == 'bgcn':
            adaptive_adj = F.dropout(self.A+self.PA, self.d, training=training)
            adaptive_adj = adaptive_adj.unsqueeze(0).expand(obs.shape[0], -1, -1)  # (batch_size, num_nodes, num_nodes)

        obs = check(obs).to(**self.tpdv)        ### sample_size, obs_dim
        obs = self.source_obs_map(obs)        ### sample_size, out_dim*max_action

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
            
        if self.supervised == 1:
            neighbor_rep = self.supervised_prediction_layer(neighbor_rep)
            neighbor_rep = torch.gather(neighbor_rep, 1, row_indices_expanded)  # (sample_size, max_action, out_dim)
            neighbor_rep[self.road_neighbor_masks[ridxs] == 0] = -1
            neighbor_rep = neighbor_rep.reshape(neighbor_rep.shape[0], -1)  # (sample_size, max_action * out_dim)
        if self.agg_type != 'none':
            obs_all_selected = torch.gather(obs_all_adaptive, 1, row_indices_expanded)  # (sample_size, max_action, out_dim)
            obs_all_selected[self.road_neighbor_masks[ridxs] == 0] = -1
            obs_all_selected = obs_all_selected.reshape(obs_all_selected.shape[0], -1)  # (sample_size, max_action * out_dim)
        obs_all_origin_selected[self.road_neighbor_masks[ridxs] == 0] = -1
        obs_all_origin_selected = obs_all_origin_selected.reshape(obs_all_origin_selected.shape[0], -1)  # (sample_size, max_action * out_dim)

        if self.agg_type != 'none':
            obs = torch.concatenate((obs, obs_all_selected), dim=-1)  # (batch_size, max_action * out_dim * 2)
        ### 用gate方式拟合
        gated_output = self.gated_layer(obs)
        obs_gated, obs_gate = gated_output.chunk(2, dim=-1)
        obs_gated = obs_gated * F.sigmoid(obs_gate)

        obs = F.relu(self.base(obs_gated))
        action_policy_values = self.action_output(obs)
        
        if self.supervised == 1:
            return action_policy_values, neighbor_rep, obs_all_origin_selected
        else:
            return action_policy_values

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

class AN_model(nn.Module):
    def __init__(self, args, source_obs_dim, obs_dim, edge_dim, max_action, roadidx2neighboridxs, device=torch.device("cpu")):
        super(AN_model, self).__init__()
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
        self.A = A
        self.d = 0.2
        self.mean_field = args.mean_field

        out_dim = 16
        self.source_obs_map = nn.Linear(source_obs_dim, out_dim*max_action)
        if args.mean_field:
            self.obs_map = nn.Linear(obs_dim, out_dim)
        else:
            self.obs_map = nn.Linear(obs_dim-1, out_dim)

        self.conv_times = 2
        self.gconv = nn.ModuleList()
        self.residual = nn.ModuleList()
        for i in range(self.conv_times):
            self.residual.append(nn.Linear(out_dim, out_dim))
            self.gconv.append(EdgeConvGat(out_dim, edge_dim))
        base_dim = out_dim * max_action * 2
        self.gated_layer = nn.Linear(base_dim, out_dim*2)

        self.base = MLPBase(args, out_dim, use_attn_internal=0, use_cat_self=True)
        self.action_output = nn.Linear(self.base.output_size, max_action)

        self.supervised_prediction_layer = nn.Linear(out_dim, out_dim)

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.to(device)

    def forward(self, obs, obs_all, edge_attrs, ridxs, corr_adj_matrix, training=True):
        adj = torch.from_numpy(self.A).unsqueeze(0).expand(obs.shape[0], -1, -1).to(**self.tpdv)  # (batch_size, num_nodes, num_nodes)
            
        obs = check(obs).to(**self.tpdv)        ### sample_size, obs_dim
        obs = self.source_obs_map(obs)        ### sample_size, out_dim*max_action

        obs_all = check(obs_all).to(**self.tpdv)        ### sample_size, num_roads, obs_dim
        edge_attrs = check(edge_attrs).to(**self.tpdv)        ### sample_size, num_roads, num_roads, 1
        if self.mean_field == 0:
            obs_all = obs_all[:, :, :-1]        ### sample_size, num_roads, obs_dim - 1
        obs_all = self.obs_map(obs_all)        ### sample_size, num_roads, out_dim
        row_indices_expanded = self.road_neighbor_idxs[ridxs].unsqueeze(2).expand(-1, -1, obs_all.shape[-1])  # (sample_size, max_action, out_dim)
        obs_all_origin_selected = torch.gather(obs_all, 1, row_indices_expanded)   # (sample_size, max_action, out_dim)

        for i in range(self.conv_times):
            obs_all_residual = self.residual[i](obs_all)
            aggregated_rep = self.gconv[i](obs_all, edge_attrs, adj)
            obs_all_adaptive = F.relu(obs_all_residual + aggregated_rep)

        obs_all_selected = torch.gather(obs_all_adaptive, 1, row_indices_expanded)  # (sample_size, max_action, out_dim)
        obs_all_selected[self.road_neighbor_masks[ridxs] == 0] = -1
        obs_all_selected = obs_all_selected.reshape(obs_all_selected.shape[0], -1)  # (sample_size, max_action * out_dim)
        obs_all_origin_selected[self.road_neighbor_masks[ridxs] == 0] = -1
        obs_all_origin_selected = obs_all_origin_selected.reshape(obs_all_origin_selected.shape[0], -1)  # (sample_size, max_action * out_dim)

        obs = torch.concatenate((obs, obs_all_selected), dim=-1)  # (batch_size, max_action * out_dim * 2)
        gated_output = self.gated_layer(obs)
        obs_gated, obs_gate = gated_output.chunk(2, dim=-1)
        obs_gated = obs_gated * F.sigmoid(obs_gate)

        obs = F.relu(self.base(obs_gated))
        action_policy_values = self.action_output(obs)
        
        return action_policy_values

def sequential_pack(layers):
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in layers:
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq

def conv2d_block(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    pad_type='zero',
    activation=None,
):
    block = []
    assert pad_type in ['zero', 'reflect', 'replication'], "invalid padding type: {}".format(pad_type)
    if pad_type == 'zero':
        pass
    elif pad_type == 'reflect':
        block.append(nn.ReflectionPad2d(padding))
        padding = 0
    elif pad_type == 'replication':
        block.append(nn.ReplicationPad2d(padding))
        padding = 0
    block.append(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=groups)
    )
    xavier_normal_(block[-1].weight)
    if activation is not None:
        block.append(activation)
    return sequential_pack(block)

class ActorModel(nn.Module):
    def __init__(self, hidden_size, action_size):
        super(ActorModel, self).__init__()
        self.name = 'actor'
        self.pre = conv2d_block(hidden_size, 6, 1)
        self.model = nn.Sequential(
            nn.Linear(action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[-1], -1).unsqueeze(-1)      # num_agents, hidden_state, 4, 1
        hidden_states = self.pre(hidden_states).squeeze(-1).sum(1)
        outputs = self.model(hidden_states)
        return outputs

class FRAP(nn.Module):
    def __init__(self, obs_size, phase_size, action_size, device):
        super(FRAP, self).__init__()
        self.device = device
        self.linear_h1 = nn.Linear(obs_size, 32)
        self.relu_h1 = nn.ReLU()
        self.linear_h2 = nn.Linear(phase_size, 32)
        self.relu_h2 = nn.ReLU()
        self.action_sizes = action_size
        self.linear = nn.Linear(64, 64)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, obs_h1_A, obs_h1_B, obs_h2):
        phase_representations = torch.zeros((obs_h1_A.shape[0], self.action_sizes, 64), dtype=torch.float32, device=self.device)
        for phase_idx in range(self.action_sizes):
            obs_1_A = self.relu_h1(self.linear_h1(obs_h1_A[:, phase_idx]))
            obs_2_A = self.relu_h1(self.linear_h2(obs_h2[:, phase_idx]))
            obs_A = self.linear(torch.cat([obs_1_A, obs_2_A], axis=1))
            obs_1_B = self.relu_h1(self.linear_h1(obs_h1_B[:, phase_idx]))
            obs_2_B = self.relu_h1(self.linear_h2(obs_h2[:, phase_idx]))
            obs_B = self.linear(torch.cat([obs_1_B, obs_2_B], axis=1))
            phase_representations[:, phase_idx, :] = obs_A + obs_B

        phase_demand_embedding_matrix = torch.zeros((obs_h1_A.shape[0], 128, self.action_sizes, self.action_sizes-1), dtype=torch.float32, device=self.device)
        for phase_idx in range(self.action_sizes):
            count = 0
            for competing_phase in range(self.action_sizes):
                if phase_idx == competing_phase:
                    continue
                phase_demand_embedding_matrix[:, :, phase_idx, count] = torch.cat([phase_representations[:, phase_idx], phase_representations[:, competing_phase]], dim=1)
                count += 1 
        phase_demand_embedding_matrix = self.bn1(self.conv1(phase_demand_embedding_matrix))
        phase_demand_embedding_matrix = self.bn2(self.conv2(phase_demand_embedding_matrix))
        phase_demand_embedding_matrix = self.conv3(phase_demand_embedding_matrix)
        return phase_demand_embedding_matrix

class ModelBody(nn.Module):
    def __init__(self, input_size, fc_layer_size, device='cpu'):
        super(ModelBody, self).__init__()
        self.name = 'model_body'

        # mlp
        self.fc_car_num = nn.Linear(input_size, fc_layer_size)
        self.act_car_num = nn.Sigmoid()
        self.fc_queue_length = nn.Linear(input_size, fc_layer_size)
        self.act_queue_length = nn.Sigmoid()
        self.fc_occupancy = nn.Linear(input_size, fc_layer_size)
        self.act_occupancy = nn.Sigmoid()
        self.fc_flow = nn.Linear(input_size, fc_layer_size)
        self.act_flow = nn.Sigmoid()
        # current phase
        self.current_phase_embedding = nn.Linear(1, fc_layer_size)
        self.current_phase_act = nn.Sigmoid()
        # self.frap = FRAP(4*fc_layer_size, fc_layer_size, 4, device)
        self.output = nn.Linear(5*fc_layer_size, 4)

    def forward(self, sa, current_phase):
        input1a = self.act_car_num(self.fc_car_num(sa[:, [0, 1, 2, 3]]))
        input2a = self.act_queue_length(self.fc_queue_length(sa[:, [4, 5, 6, 7]]))
        input3a = self.act_occupancy(self.fc_occupancy(sa[:, [8, 9, 10, 11]]))
        input4a = self.act_flow(self.fc_flow(sa[:, [12, 13, 14, 15]]))

        current_phase = current_phase.reshape(current_phase.shape[0], 4, 1)
        current_phase = self.current_phase_act(self.current_phase_embedding(current_phase))

        inputa = torch.cat([input1a, input2a, input3a, input4a], dim=1)
        inputa = inputa[:, [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]]
        inputa = inputa.reshape(inputa.shape[0], 4, -1)

        phase_scores = self.output(torch.cat([inputa, current_phase], dim=2))
        return phase_scores


class GESA(nn.Module):
    def __init__(self, action_size, device):
        super(GESA, self).__init__()
        hidden_layer_size = 32
        self.device = device
        self.body_model = ModelBody(1, hidden_layer_size, device=device).to(device)
        self.actor_model = ActorModel(4, action_size).to(device)

    def forward(self, sa, current_phase):
        sa = check(sa).to(self.device)
        current_phase = check(current_phase).to(self.device)

        hidden_states = self.body_model(sa.float(), current_phase.float())
        logits = self.actor_model(hidden_states)
        return logits