import os
import torch
from tap import Tap
from typing import List

import random
import numpy as np


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


    
class ArgumentParser(Tap):
    # experiment part
    part_name: str = "LGMformer"
    dataset: str = 'ogbn-arxiv'
    batch_size: int = 1024
    test_batch_size: int = 1024
    sizes: List = [25]
    test_sizes: List = [25]
    # sizes: List = [15, 10, 5]
    # test_sizes: List = [15, 10, 5]

    sample_hop_size: int = 40
    
    num_workers: int = 4
    hetero_train_prop: float = 0.5
    test_freq: int = 2
    device: int = 0
    weight_decay: float = 1e-05
    warmup_epochs: int = 200
    peak_lr: float = 5e-4
    end_lr: float = 1e-9
    
    h_times: int = 1
    seed: int = 0
    
    num_centroids: int = 4096
    max_patience: int = 30
    

    feature_hops: int = 5
    num_heads: int = 8
    pe_dim: int = 15
    hidden_dim: int = 128
    global_dim: int = 128
    
    epochs: int = 2000
    test_start_epoch: int = 0
    num_layers: int = 1
    edge_former_num_layers: int = 1
    conv_type: str = 'full'
    token_type: str = 'full'
    
    attn_dropout: float = 0.5
    ff_dropout: float = 0.5
    is_acc: bool = True
    
    need_sample: bool = False
    undirected: bool = False
    
    splits_idx: int = 0
    compare_valid_acc: int = 0
    
    num_prototypes_per_class: int = 3
    
    save_ckpt: bool = True
    
    pos_enc_type: str = "none"

    knn_graph_hops: int = 30
    knn_graph_pos_hops: int = 2

    root_data_path: str = '/nfs/lwd/Codes/GOAT'
    root_other_path: str = '/nfs/lwd/Codes/GOAT'
    
    data_root = f'{root_data_path}/data/'
    pos_enc_path = f'{root_other_path}/pos_enc/'
    
    node_features_save_path = f'{root_other_path}/output/re_features/'
    save_checkpoint_path = f'{root_other_path}/output/checkpoint/'
    save_acc_loss_dict = f'{root_other_path}/output/acc_loss_dict/'
    samplers_save_path = f'{root_other_path}/output/samplers/'
    
    def process_args(self):
        self.process_path()

    def process_path(self):
        if not os.path.exists(self.root_data_path):
            os.makedirs(self.root_data_path)
        if not os.path.exists(self.root_other_path):
            os.makedirs(self.root_other_path)
        if not os.path.exists(self.node_features_save_path):
            os.makedirs(self.node_features_save_path)
        if not os.path.exists(self.pos_enc_path):
            os.makedirs(self.pos_enc_path)
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        if not os.path.exists(self.save_checkpoint_path):
            os.makedirs(self.save_checkpoint_path)
        if not os.path.exists(self.save_acc_loss_dict):
            os.makedirs(self.save_acc_loss_dict)
        if not os.path.exists(self.samplers_save_path):
            os.makedirs(self.samplers_save_path)

def get_params():
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    args = ArgumentParser().parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    if args.dataset in {'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions', 'wikics'}:
        args.seed = 0
    if args.dataset in {'amazon-ratings', 'wikics'}:
        args.compare_valid_acc = 1
    else:
        args.compare_valid_acc = 0
        
    args.test_sizes[0] = args.sample_hop_size
    args.sizes[0] = args.sample_hop_size
    set_seed(args.seed)
    return args

