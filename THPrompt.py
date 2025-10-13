import torch
import torch.nn as nn
import numpy as np


class EdgePromptGenerator(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=64):
        super().__init__()
        self.node_aggregator = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_aggregator = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim

        self.prompt_generator = nn.Linear(1, edge_feat_dim)
        
    def aggregate_features(self, features, aggregator, device):
        if len(features) == 0:
            return torch.zeros(self.hidden_dim, device=device)
        
        features = [f.to(device) for f in features]
        return aggregator(torch.stack(features).mean(dim=0))
        
    def forward(self, u, v, t1, t2, t3, 
                node_features, edge_features,
                node_timestamps, edge_timestamps,
                node_history, edge_history,node_time_varying):
        device = node_features.device

        if node_time_varying:
            input_dim = 2 * (self.hidden_dim + self.hidden_dim)
        else:
            input_dim = 2 * (self.hidden_dim + self.node_feat_dim)
        
        if self.prompt_generator.in_features != input_dim:
            self.prompt_generator = nn.Linear(input_dim, self.edge_feat_dim).to(device)
        
        if not node_time_varying:
            u1 = node_features[u]
        else:
            u_ts = node_timestamps.get(u, [])
            u_feat_indices = node_history.get(u, [])
            valid_feats = []
            for ts, idx in zip(u_ts, u_feat_indices):
                if t2 <= ts <= t1:
                    valid_feats.append(node_features[idx])
            u1 = self.aggregate_features(valid_feats, self.node_aggregator, device)

        u_edge_ts = torch.tensor(edge_timestamps[u], device=t3.device)
        u_edge_mask = (u_edge_ts >= t3) & (u_edge_ts <= t1)
        u_edge_feats = edge_features[edge_history[u]][u_edge_mask]
        u2 = self.aggregate_features(u_edge_feats, self.edge_aggregator, device)
        
        u3 = torch.cat([u1, u2], dim=0)       

        if not node_time_varying:
            v1 = node_features[v]
        else:
            v_ts = node_timestamps.get(v, [])
            v_feat_indices = node_history.get(v, [])
            valid_feats = []
            for ts, idx in zip(v_ts, v_feat_indices):
                if t2 <= ts <= t1:
                    valid_feats.append(node_features[idx])
            
            v1 = self.aggregate_features(valid_feats, self.node_aggregator, device)
        
        v_edge_ts = torch.tensor(edge_timestamps[v], device=t3.device)
        v_edge_mask = (v_edge_ts >= t3) & (v_edge_ts <= t1)
        v_edge_feats = edge_features[edge_history[v]][v_edge_mask]
        v2 = self.aggregate_features(v_edge_feats, self.edge_aggregator, device)
        
        v3 = torch.cat([v1, v2], dim=0)
        edge_prompt = self.prompt_generator(torch.cat([u3, v3], dim=0))
        return edge_prompt
    

class task_prompt_layer(nn.Module):
    def __init__(self,input_dim):
        super(task_prompt_layer, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, node_embedding):
        node_embedding=node_embedding*self.weight
        return node_embedding


class structure_prompt_layer(nn.Module):
    def __init__(self,size,input_dim):
        super(structure_prompt_layer, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(size,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self ,id, node_embedding):
        node_embedding=node_embedding + self.weight[id]
        return node_embedding