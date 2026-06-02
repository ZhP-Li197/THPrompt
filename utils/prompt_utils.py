import numpy as np
import torch


def compute_node_degrees_at_time(neighbor_finder, nodes, timestamps):
    """Compute the degree of each node at its given timestamp.

    Args:
        neighbor_finder: NeighborFinder instance with find_before method
        nodes: (B,) numpy array of node IDs
        timestamps: (B,) numpy array of timestamps

    Returns:
        degrees: (B, 1) tensor of degree counts
    """
    degrees = []
    for node, ts in zip(nodes, timestamps):
        neighbors, _, _ = neighbor_finder.find_before(node, ts)
        degrees.append(len(neighbors))
    return torch.tensor(degrees, dtype=torch.float32).unsqueeze(1)


def compute_graph_level_features(memory, n_nodes):
    """Compute graph-level feature by mean pooling all node memory states.

    Args:
        memory: (N, d) memory tensor for all nodes
        n_nodes: number of actual nodes (excluding padding node 0)

    Returns:
        g_t: (d,) graph-level feature vector
    """
    if n_nodes <= 1:
        return memory[0]
    return memory[1:n_nodes + 1].mean(dim=0)


def compute_avg_degree_at_time(neighbor_finder, n_nodes, timestamp):
    """Compute the average node degree at a given timestamp.

    Args:
        neighbor_finder: NeighborFinder instance
        n_nodes: total number of nodes
        timestamp: the time point to evaluate

    Returns:
        d_bar: scalar average degree
    """
    total_degree = 0
    count = 0
    sample_size = min(n_nodes, 200)
    sampled_nodes = np.random.choice(range(1, n_nodes + 1), sample_size, replace=False)
    for node in sampled_nodes:
        neighbors, _, _ = neighbor_finder.find_before(node, timestamp)
        total_degree += len(neighbors)
        count += 1
    return total_degree / max(count, 1)
