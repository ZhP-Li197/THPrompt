import math
import time
import sys
import pickle
from pathlib import Path

import torch
import numpy as np

from TGN.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics
from utils.evaluation import eval_edge_prediction
from config import get_args

torch.manual_seed(0)
np.random.seed(0)

config = get_args()

MODEL_SAVE_PATH = f'./saved_models/{config.prefix}-{config.data}.pth'
RESULTS_PATH = f"results/{config.prefix}.pkl"
Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
Path("results/").mkdir(parents=True, exist_ok=True)

get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{config.prefix}-{config.data}-{epoch}.pth'

node_features, edge_features, full_data, train_data, val_data, test_data, \
new_node_val_data, new_node_test_data = get_data(
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

for run in range(config.n_runs):
    if config.n_runs > 1:
        RESULTS_PATH = f"results/{config.prefix}_{run}.pkl"
    
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
        dyrep=config.dyrep
    )
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=config.lr)
    tgn.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / config.bs)

    val_aps = []
    new_nodes_val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []

    early_stopper = EarlyStopMonitor(max_round=config.patience)
    
    for epoch in range(config.n_epoch):
        start_epoch_time = time.time()
        
        if config.use_memory:
            tgn.memory.__init_memory__()
        
        tgn.set_neighbor_finder(train_neighbor_finder)
        mean_epoch_loss = []
        
        for batch_start in range(0, num_batch, config.backprop_every):
            loss = 0
            optimizer.zero_grad()
            
            for j in range(config.backprop_every):
                batch_idx = batch_start + j
                if batch_idx >= num_batch:
                    continue

                start_idx = batch_idx * config.bs
                end_idx = min(num_instance, start_idx + config.bs)
                
                sources_batch = train_data.sources[start_idx:end_idx]
                destinations_batch = train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]
                
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
                
                loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

            loss /= config.backprop_every
            loss.backward()
            optimizer.step()
            mean_epoch_loss.append(loss.item())
            
            if config.use_memory:
                tgn.memory.detach_memory()
        
        epoch_times.append(time.time() - start_epoch_time)
        
        tgn.set_neighbor_finder(full_neighbor_finder)

        if config.use_memory:
            train_memory_backup = tgn.memory.backup_memory()
        
        val_ap, val_auc = eval_edge_prediction(
            model=tgn,
            negative_edge_sampler=val_rand_sampler,
            data=val_data,
            n_neighbors=config.n_degree
        )
        
        if config.use_memory:
            val_memory_backup = tgn.memory.backup_memory()
            tgn.memory.restore_memory(train_memory_backup)

        nn_val_ap, nn_val_auc = eval_edge_prediction(
            model=tgn,
            negative_edge_sampler=nn_val_rand_sampler,
            data=new_node_val_data,
            n_neighbors=config.n_degree
        )
        
        if config.use_memory:
            tgn.memory.restore_memory(val_memory_backup)
        
        val_aps.append(val_ap)
        new_nodes_val_aps.append(nn_val_ap)
        train_losses.append(np.mean(mean_epoch_loss))
        total_epoch_times.append(time.time() - start_epoch_time)

        pickle.dump({
            "val_aps": val_aps,
            "new_nodes_val_aps": new_nodes_val_aps,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(RESULTS_PATH, "wb"))
        
        if early_stopper.early_stop_check(val_ap):
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            tgn.load_state_dict(torch.load(best_model_path))
            tgn.eval()
            break
        else:
            torch.save(tgn.state_dict(), get_checkpoint_path(epoch))
    
    if config.use_memory:
        val_memory_backup = tgn.memory.backup_memory()
    
    tgn.embedding_module.neighbor_finder = full_neighbor_finder
    test_ap, test_auc = eval_edge_prediction(
        model=tgn,
        negative_edge_sampler=test_rand_sampler,
        data=test_data,
        n_neighbors=config.n_degree
    )
    
    if config.use_memory:
        tgn.memory.restore_memory(val_memory_backup)
    
    nn_test_ap, nn_test_auc = eval_edge_prediction(
        model=tgn,
        negative_edge_sampler=nn_test_rand_sampler,
        data=new_node_test_data,
        n_neighbors=config.n_degree
    )
    
    print(f'Test statistics: Old nodes -- auc: {test_auc:.4f}, ap: {test_ap:.4f}')
    print(f'Test statistics: New nodes -- auc: {nn_test_auc:.4f}, ap: {nn_test_ap:.4f}')
    
    pickle.dump({
        "val_aps": val_aps,
        "new_nodes_val_aps": new_nodes_val_aps,
        "test_ap": test_ap,
        "new_node_test_ap": nn_test_ap,
        "epoch_times": epoch_times,
        "train_losses": train_losses,
        "total_epoch_times": total_epoch_times
    }, open(RESULTS_PATH, "wb"))
    
    if config.use_memory:
        tgn.memory.restore_memory(val_memory_backup)
    
    torch.save(tgn.state_dict(), MODEL_SAVE_PATH)