import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TopologyAwarePrompt(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, rank=16,
                 n_bases=8, local_hidden_dim=64):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.rank = rank
        self.n_bases = n_bases

        # Low-rank factor matrices U, V ∈ R^{d×r}
        self.U = nn.Parameter(torch.empty(node_feat_dim, rank))
        self.V = nn.Parameter(torch.empty(node_feat_dim, rank))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

        # Local-to-global refinement
        # r_local = [x_i ∥ x_j ∥ d_i ∥ d_j] → dim = 2*d + 2
        local_input_dim = 2 * node_feat_dim + 2
        self.local_mlp = nn.Sequential(
            nn.Linear(local_input_dim, local_hidden_dim),
            nn.ReLU(),
            nn.Linear(local_hidden_dim, local_hidden_dim),
        )
        self.W_G = nn.Linear(node_feat_dim + 1, local_hidden_dim, bias=False)

        # Hybrid fusion + basis decomposition
        # s_topo = [s_LR ∥ s_LG], dim = rank + local_hidden_dim
        topo_context_dim = rank + local_hidden_dim
        self.topo_bases = nn.Parameter(torch.empty(n_bases, edge_feat_dim))
        self.topo_attn_keys = nn.Parameter(torch.empty(n_bases, topo_context_dim))
        nn.init.xavier_uniform_(self.topo_bases)
        nn.init.xavier_uniform_(self.topo_attn_keys)

        self.score_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_feat_dim // 2),
            nn.ReLU(),
            nn.Linear(edge_feat_dim // 2, 1),
        )

    def forward(self, x_i, x_j, d_i, d_j, g_t, d_bar):
        """
	Batched topology-aware prompt generation.
        """
        B = x_i.size(0)
        device = x_i.device

        # --- Low-rank generative prior ---
        u_i = x_i @ self.U  # (B, r)
        v_j = x_j @ self.V  # (B, r)
        s_lr = u_i * v_j     # (B, r) — element-wise (Hadamard)

        # --- Local structural representation ---
        r_local_ij = torch.cat([x_i, x_j, d_i, d_j], dim=1)  # (B, 2d+2)
        r_local_ji = torch.cat([x_j, x_i, d_j, d_i], dim=1)  # (B, 2d+2)

        # --- Global representation ---
        d_bar_t = torch.tensor([d_bar], device=device, dtype=x_i.dtype)
        r_global = torch.cat([g_t, d_bar_t], dim=0)  # (d+1,)

        # --- Local-to-global refinement ---
        s_lg = (self.local_mlp(r_local_ij)
                + self.local_mlp(r_local_ji)
                + self.W_G(r_global).unsqueeze(0))  # (B, local_hidden)

        # --- Hybrid fusion via basis decomposition ---
        s_topo = torch.cat([s_lr, s_lg], dim=1)  # (B, rank + local_hidden)
        logits = s_topo @ self.topo_attn_keys.t()  # (B, K1)
        alpha = F.softmax(logits, dim=1)           # (B, K1)
        p_topo = alpha @ self.topo_bases            # (B, F_E)

        return p_topo

    def compute_topo_loss(self, p_pos, p_neg):
        y_pos = torch.sigmoid(self.score_mlp(p_pos)).squeeze(-1)
        y_neg = torch.sigmoid(self.score_mlp(p_neg)).squeeze(-1)
        loss = (-torch.log(y_pos + 1e-8).sum()
                - torch.log(1 - y_neg + 1e-8).sum())
        return loss / (y_pos.size(0) + y_neg.size(0))


class HistoryAwarePrompt(nn.Module):
    def __init__(self, node_feat_dim, n_bases=8, n_neighbors=10, K=5):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.n_bases = n_bases
        self.n_neighbors = n_neighbors
        self.K = K

        his_context_dim = 2 * node_feat_dim

        self.his_bases = nn.Parameter(torch.empty(n_bases, node_feat_dim))
        self.his_attn_keys = nn.Parameter(torch.empty(n_bases, his_context_dim))
        nn.init.xavier_uniform_(self.his_bases)
        nn.init.xavier_uniform_(self.his_attn_keys)

    def forward(self, memory, source_nodes, timestamps, neighbor_finder,
                memory_module=None):
        B = len(source_nodes)
        device = memory.device
        d = memory.size(1)

        # --- Self-history context ---
        # Use current memory state as summary of self-history
        x_self = memory[source_nodes]  # (B, d)

        if memory_module is not None and hasattr(memory_module, 'messages'):
            for idx, node_id in enumerate(source_nodes):
                msgs = memory_module.messages.get(node_id, [])
                if len(msgs) > 0:
                    recent_msgs = msgs[-self.K:]
                    msg_tensors = torch.stack([m[0] for m in recent_msgs])
                    msg_mean = msg_tensors.mean(dim=0)
                    if msg_mean.shape[0] == d:
                        x_self[idx] = (x_self[idx] + msg_mean) / 2.0

        # --- Neighborhood context ---
        # Reuse neighbor_finder to get recent neighbors and their memory states
        neighbors, _, _ = neighbor_finder.get_temporal_neighbor(
            source_nodes, timestamps, n_neighbors=self.n_neighbors
        )
        neighbors_flat = neighbors.flatten()
        neigh_memory = memory[neighbors_flat].view(B, self.n_neighbors, d)

        neigh_mask = torch.from_numpy(neighbors).long().to(device) != 0  # (B, n_neighbors)
        neigh_mask = neigh_mask.unsqueeze(-1).float()  # (B, n_neighbors, 1)
        neigh_sum = (neigh_memory * neigh_mask).sum(dim=1)
        neigh_count = neigh_mask.sum(dim=1).clamp(min=1)
        x_neigh = neigh_sum / neigh_count  # (B, d) — mean pooling readout

        # --- History-aware context ---
        s_his = torch.cat([x_self, x_neigh], dim=1)  # (B, 2d)

        # --- Basis decomposition ---
        logits = s_his @ self.his_attn_keys.t()  # (B, K2)
        beta = F.softmax(logits, dim=1)           # (B, K2)
        q_his = beta @ self.his_bases              # (B, F_0)

        return q_his


class TaskPromptLayer(nn.Module):
    """Task-specific element-wise scaling prompt."""

    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, input_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, node_embedding):
        return node_embedding * self.weight
