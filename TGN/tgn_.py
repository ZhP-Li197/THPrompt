import logging
import numpy as np
import torch
from collections import defaultdict
from utils.utils import MergeLayer
from TGN.memory import Memory
from TGN.message_aggregator import get_message_aggregator
from TGN.message_function import get_message_function
from TGN.memory_updater import get_memory_updater
from TGN.embedding_module import get_embedding_module
from TGN.time_encoding import TimeEncode
from THPrompt import TopologyAwarePrompt, HistoryAwarePrompt, TaskPromptLayer
from utils.prompt_utils import (compute_node_degrees_at_time,
                                compute_graph_level_features,
                                compute_avg_degree_at_time)
import torch.nn as nn


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False, struc_prompt_tag=False,
               node_feat_dim=None, edge_feat_dim=None,
               rank=16, n_topo_bases=8, n_his_bases=8, self_history_K=5,
               topo_lambda=0.1):
    super(TGN, self).__init__()
    self.struc_prompt_tag = struc_prompt_tag

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.n_layers = n_layers

    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep
    self.topo_lambda = topo_lambda

    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.memory = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    if self.use_memory:
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads,
                                                 dropout=dropout,
                                                 use_memory=use_memory)

    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                     self.n_node_features, 1)

    # --- Topology-Aware Generative Prompt ---
    _node_feat_dim = node_feat_dim or self.n_node_features
    _edge_feat_dim = edge_feat_dim or self.n_edge_features
    self.topo_prompt = TopologyAwarePrompt(
        node_feat_dim=_node_feat_dim,
        edge_feat_dim=_edge_feat_dim,
        rank=rank,
        n_bases=n_topo_bases,
    )

    # --- History-Aware Temporal Context Prompt ---
    self.history_prompt = HistoryAwarePrompt(
        node_feat_dim=_node_feat_dim,
        n_bases=n_his_bases,
        n_neighbors=n_neighbors or 10,
        K=self_history_K,
    )

    # Task prompt layer
    self.task_prompt = TaskPromptLayer(_edge_feat_dim)

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, n_neighbors=20):
    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0)

    # Compute base node embeddings (frozen backbone)
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    if memory is not None:
      his_prompt_src = self.history_prompt(
          memory, source_nodes, edge_times,
          self.neighbor_finder, self.memory
      )
      his_prompt_dst = self.history_prompt(
          memory, destination_nodes, edge_times,
          self.neighbor_finder, self.memory
      )
      his_prompt_neg = self.history_prompt(
          memory, negative_nodes, edge_times,
          self.neighbor_finder, self.memory
      )
      source_node_embedding = source_node_embedding + his_prompt_src
      destination_node_embedding = destination_node_embedding + his_prompt_dst
      negative_node_embedding = negative_node_embedding + his_prompt_neg

    if self.use_memory:
      if self.memory_update_at_start:
        self.update_memory(positives, self.memory.messages)
        assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-1), \
          "Something wrong in how the memory was updated"
        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs,
                                                                    memory=memory)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs,
                                                                              memory=memory)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        negative_node_embedding = memory[negative_nodes]

    return source_node_embedding, destination_node_embedding, negative_node_embedding

  def _compute_topo_prompts_batched(self, source_nodes, destination_nodes, edge_times, memory):
    B = len(source_nodes)

    x_src = memory[source_nodes]  # (B, d)
    x_dst = memory[destination_nodes]  # (B, d)

    d_src = compute_node_degrees_at_time(
        self.neighbor_finder, source_nodes, edge_times
    ).to(self.device)  # (B, 1)
    d_dst = compute_node_degrees_at_time(
        self.neighbor_finder, destination_nodes, edge_times
    ).to(self.device)  # (B, 1)

    g_t = compute_graph_level_features(memory, self.n_nodes - 1)  # (d,)
    max_ts = edge_times.max() if len(edge_times) > 0 else 0.0
    d_bar = compute_avg_degree_at_time(self.neighbor_finder, self.n_nodes - 1, max_ts)

    p_topo = self.topo_prompt(x_src, x_dst, d_src, d_dst, g_t, d_bar)
    return p_topo

  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, n_neighbors=20):
    n_samples = len(source_nodes)

    source_node_embedding, destination_node_embedding, negative_node_embedding = \
        self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

    score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding,
                                           negative_node_embedding])).squeeze(dim=0)
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]

    return pos_score.sigmoid(), neg_score.sigmoid()

  def compute_topo_loss(self, source_nodes, destination_nodes, negative_nodes,
                        edge_times, memory):
    """
	Compute topology-aware loss.
    """
    if memory is None:
      return torch.tensor(0.0, device=self.device)

    # Positive edge prompts
    p_pos = self._compute_topo_prompts_batched(
        source_nodes, destination_nodes, edge_times, memory
    )
    # Negative edge prompts
    p_neg = self._compute_topo_prompts_batched(
        source_nodes, negative_nodes, edge_times, memory
    )
    return self.topo_prompt.compute_topo_loss(p_pos, p_neg)

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs, memory=None):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    if memory is not None:
      p_topo = self._compute_topo_prompts_batched(
          source_nodes, destination_nodes, edge_times.cpu().numpy(), memory
      )
      edge_features = edge_features + p_topo  # Eq. 10: additive prompt

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    if not self.use_destination_embedding_in_message:
      destination_memory = self.memory.get_memory(destination_nodes)
    else:
      destination_memory = destination_node_embedding.detach()

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def update_memory(self, nodes, messages):
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(nodes, messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_memory(self, nodes, messages):
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(nodes, messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(
        unique_nodes, unique_messages, timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
