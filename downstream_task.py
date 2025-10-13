import math
import time
import sys
import pickle
from pathlib import Path

import torch
import numpy as np

from TGN.tgn_ import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_d_data, compute_time_statistics
from utils.evaluation import eval_edge_prediction_fewshot, eval_edge_prediction
from config import get_args

torch.manual_seed(0)
np.random.seed(0)

config = get_args()

MODEL_SAVE_PATH = f'./saved_models/{config.prefix}-{config.data}.pth'
Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./results/").mkdir(parents=True, exist_ok=True)

node_features, edge_features, full_data, train_data, val_data, test_data, \
new_node_val_data, new_node_test_data = get_d_data(
    config.data,
    different_new_nodes_between_val_and_test=False,
    randomize_features=False
)

train_neighbor_finder = get_neighbor_finder(train_data, config.uniform)
full_neighbor_finder = get_neighbor_finder(full_data, config.uniform)

train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')

mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

test_aucs = []
test_nn_aucs = []

for task in range(1):
    tgn = TGN(
        neighbor_finder=train_neighbor_finder,
        node_features=node_features,
        edge_features=edge_features,
        device=device,
        n_layers=config.n_layer,
        n_heads=config.n_head,
        dropout=config.drop_out,
        use_memory=config.use_memory,
        message_dimension=config.message_dim,
        memory_dimension=config.memory_dim,
        memory_update_at_start=not config.memory_update_at_end,
        embedding_module_type=config.embedding_module,
        message_function=config.message_function,
        aggregator_type=config.aggregator,
        memory_updater_type=config.memory_updater,
        n_neighbors=config.n_degree,
        mean_time_shift_src=mean_time_shift_src,
        std_time_shift_src=std_time_shift_src,
        mean_time_shift_dst=mean_time_shift_dst,
        std_time_shift_dst=std_time_shift_dst,
        use_destination_embedding_in_message=config.use_destination_embedding_in_message,
        use_source_embedding_in_message=config.use_source_embedding_in_message,
        dyrep=config.dyrep,
        struc_prompt_tag=True,
        node_feat_dim=172,
        edge_feat_dim=172,
        t2_ratio=0.8,
        t3_ratio=0.8
    )
    
    criterion = torch.nn.BCELoss()
    tgn.load_state_dict(torch.load(MODEL_SAVE_PATH), strict=False)
    tgn.to(device)

    optimizer = torch.optim.Adam(tgn.affinity_score.parameters(), lr=config.lr)
    struc_prompt_optimizer = torch.optim.Adam(tgn.struc_prompt.parameters(), lr=0.01)
    
    for epoch in range(config.n_epoch):
        start_epoch_time = time.time()
        
        if config.use_memory:
            tgn.memory.__init_memory__()
        
        tgn.set_neighbor_finder(train_neighbor_finder)
        
        optimizer.zero_grad()
        struc_prompt_optimizer.zero_grad()

        sample_size = min(10, len(train_data.sources))
        train_indices = np.random.choice(len(train_data.sources), sample_size, replace=False)
        
        sources_batch = train_data.sources[train_indices]
        destinations_batch = train_data.destinations[train_indices]
        edge_idxs_batch = train_data.edge_idxs[train_indices]
        timestamps_batch = train_data.timestamps[train_indices]
        
        size = len(sources_batch)
        _, negatives_batch = train_rand_sampler.sample(size)
        
        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)
        
        tgn.train()
        pos_prob, neg_prob = tgn.compute_edge_probabilities(
            sources_batch, destinations_batch, negatives_batch,
            timestamps_batch, edge_idxs_batch, config.n_degree
        )
        
        src_prompt = tgn.struc_prompt.weight[sources_batch]
        dst_prompt = tgn.struc_prompt.weight[destinations_batch]
        pos_prompt_sim = torch.sum(src_prompt * dst_prompt, dim=1)
        pos_prompt_prob = torch.sigmoid(pos_prompt_sim)

        neg_dst = torch.randint(0, tgn.struc_prompt.weight.size(0), (size,), device=device)
        neg_src_prompt = tgn.struc_prompt.weight[sources_batch]
        neg_dst_prompt = tgn.struc_prompt.weight[neg_dst]
        neg_prompt_sim = torch.sum(neg_src_prompt * neg_dst_prompt, dim=1)
        neg_prompt_prob = torch.sigmoid(neg_prompt_sim)

        prompt_loss = 0.5 * criterion(pos_prompt_prob, pos_label) + 0.5 * criterion(neg_prompt_prob, neg_label)
        total_loss = criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label) + prompt_loss
        total_loss /= config.backprop_every
        
        total_loss.backward()
        optimizer.step()
        struc_prompt_optimizer.step()
        
        if config.use_memory:
            tgn.memory.detach_memory()

        tgn.set_neighbor_finder(full_neighbor_finder)
        
        train_memory_backup = None
        if config.use_memory:
            train_memory_backup = tgn.memory.backup_memory()

        _, val_auc = eval_edge_prediction_fewshot(
            model=tgn,
            negative_edge_sampler=val_rand_sampler,
            data=val_data,
            n_neighbors=config.n_degree
        )
        
        val_memory_backup = None
        if config.use_memory:
            val_memory_backup = tgn.memory.backup_memory()
            tgn.memory.restore_memory(train_memory_backup)
        
        _, nn_val_auc = eval_edge_prediction(
            model=tgn,
            negative_edge_sampler=nn_val_rand_sampler,
            data=new_node_val_data,
            n_neighbors=config.n_degree
        )
        
        if config.use_memory:
            tgn.memory.restore_memory(val_memory_backup)
        
        epoch_duration = time.time() - start_epoch_time
        print(f'Epoch: {epoch}, Time: {epoch_duration:.2f}s, Loss: {total_loss.item():.4f}')
        print(f'Val AUC: {val_auc:.4f}, New Node Val AUC: {nn_val_auc:.4f}')

    tgn.embedding_module.neighbor_finder = full_neighbor_finder
    _, test_auc = eval_edge_prediction(
        model=tgn,
        negative_edge_sampler=test_rand_sampler,
        data=test_data,
        n_neighbors=config.n_degree
    )
    test_aucs.append(test_auc)
    
    if config.use_memory:
        tgn.memory.restore_memory(val_memory_backup)
    
    _, nn_test_auc = eval_edge_prediction(
        model=tgn,
        negative_edge_sampler=nn_test_rand_sampler,
        data=new_node_test_data,
        n_neighbors=config.n_degree
    )
    test_nn_aucs.append(nn_test_auc)
    
    print(f'\nTest statistics: Old nodes -- AUC: {test_auc:.4f}')
    print(f'Test statistics: New nodes -- AUC: {nn_test_auc:.4f}')

output_folder = "./"
np.savetxt(f"{output_folder}/{config.name}_total_mean_aucs.txt", [sum(test_aucs) / len(test_aucs)], fmt='%.4f')
np.savetxt(f"{output_folder}/{config.name}_nn_total_mean_aucs.txt", [sum(test_nn_aucs) / len(test_nn_aucs)], fmt='%.4f')
