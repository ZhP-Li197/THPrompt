# config.py
import argparse
import sys

def get_args():
    """
    Defines and parses command-line arguments for the TGN training script.
    """
    arg_parser = argparse.ArgumentParser('TGN Self-Supervised Training')
    
    # Data and model paths
    arg_parser.add_argument('-d', '--data', type=str, help='Dataset name (e.g., wikipedia or reddit)', default='wikipedia')
    arg_parser.add_argument('--prefix', type=str, default='tgn_model', help='Prefix for naming checkpoints and models')
    arg_parser.add_argument('--name', type=str, default='TGN_WIKI', help='Prefix for naming the result file')

    # Training settings
    arg_parser.add_argument('--bs', type=int, default=100, help='Batch size')
    arg_parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    arg_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    arg_parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    arg_parser.add_argument('--n_runs', type=int, default=1, help='Number of runs for averaging results')
    arg_parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    arg_parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
    arg_parser.add_argument('--backprop_every', type=int, default=1, help='Backpropagate every N batches')
    
    # TGN Model parameters
    arg_parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    arg_parser.add_argument('--n_head', type=int, default=2, help='Number of attention heads')
    arg_parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    arg_parser.add_argument('--node_dim', type=int, default=100, help='Node embedding dimensions')
    arg_parser.add_argument('--time_dim', type=int, default=100, help='Time embedding dimensions')
    arg_parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=["graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    arg_parser.add_argument('--message_function', type=str, default="identity", choices=["mlp", "identity"], help='Type of message function')
    arg_parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
    arg_parser.add_argument('--use_destination_embedding_in_message', action='store_true', help='Whether to use the embedding of the destination node as part of the message')
    arg_parser.add_argument('--use_source_embedding_in_message', action='store_true', help='Whether to use the embedding of the source node as part of the message')
    
    # Memory module parameters
    arg_parser.add_argument('--use_memory', action='store_true', help='Use a node memory module')
    arg_parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"], help='Type of memory updater')
    arg_parser.add_argument('--memory_update_at_end', action='store_true', help='Update memory at the end of the batch')
    arg_parser.add_argument('--message_dim', type=int, default=100, help='Message dimensions')
    arg_parser.add_argument('--memory_dim', type=int, default=172, help='Memory dimensions for each user')

    # Few-shot learning parameters
    arg_parser.add_argument('--train_shot_num', type=int, default=3, help='Number of few-shot training samples')
    arg_parser.add_argument('--val_shot_num', type=int, default=3, help='Number of few-shot validation samples')
    arg_parser.add_argument('--test_shot_num', type=int, default=100, help='Number of few-shot test samples')

    # Other options
    arg_parser.add_argument('--uniform', action='store_true', help='Use uniform sampling for temporal neighbors')
    arg_parser.add_argument('--new_node', action='store_true', help='Activate new node modeling')
    arg_parser.add_argument('--dyrep', action='store_true', help='Run the DyRep model variant')
    arg_parser.add_argument('--tag', type=int, default=3, help='A general-purpose tag for the model run')

    try:
        config = arg_parser.parse_args()
        return config
    except:
        arg_parser.print_help()
        sys.exit(0)