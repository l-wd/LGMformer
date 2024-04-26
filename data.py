import os
import torch
import scipy
import scipy.io
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch_geometric.transforms as T

from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Coauthor, WikiCS
from google_drive_downloader import GoogleDriveDownloader as gdd

import sys
sys.path.append('.')
sys.path.append("..")

dataset_drive_url = {
    'twitch-gamer_feat' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia', 
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ', 
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M 
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M 
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
}

def get_dataset(args, dataset_name, data_root, hetero_train_prop=0.5):
    
    if dataset_name.startswith('ogbn'):
        data, num_classes, split_idx, x, y = load_ogbn_dataset(args, data_root, dataset_name)
    elif dataset_name == 'arxiv-year':
        data, num_classes, split_idx, x, y = load_arxiv_year_dataset(args, data_root)
    elif dataset_name == 'pokec':
        data, num_classes, split_idx, x, y = load_pokec_mat(args, data_root)
    elif dataset_name == 'genius':
        data, num_classes, split_idx, x, y = load_genius(args, data_root)
    elif dataset_name == 'snap-patents':
        data, num_classes, split_idx, x, y = load_snap_patents_mat(args, data_root)
    elif dataset_name == 'twitch-gamer':
        data, num_classes, split_idx, x, y = load_twitch_gamer_dataset(args, data_root)
    elif dataset_name in {'cora', 'citeseer', "computer", "photo", 'pubmed'}:
        data_path = f'{data_root}/NAGformer_small/'

        file_path = data_path+dataset_name+".pt"

        data_list = torch.load(file_path)

        adj = data_list[0]
        features = data_list[1]
        labels = data_list[2]

        idx_train = data_list[3]
        idx_val = data_list[4]
        idx_test = data_list[5]

        adj = adj._indices()
        num_nodes = features.shape[0]
        
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels)
        idx_train = torch.tensor(idx_train)
        idx_val = torch.tensor(idx_val)
        idx_test = torch.tensor(idx_test)

          
        class MyObject:
            pass
        data = MyObject()
        x = data.x = features
        y = data.y = torch.tensor(labels)
        data.num_features = data.x.shape[-1]
        data.edge_index = adj
        data.num_nodes = num_nodes

        num_classes = labels.max().item() + 1
        
        split_idx = {'train': idx_train, 'valid': idx_val, 'test': idx_test}
    elif dataset_name == "cs":
        dataset_dir = f'{data_root}/NAGformer_small/'
        dataset = Coauthor(dataset_dir, name='CS')
        data_o = dataset[0]
          
        edge_index = data_o.edge_index
        node_feat = data_o.x
        label = data_o.y
        num_nodes = data_o.num_nodes

        class MyObject:
            pass
        data = MyObject()
        x = data.x = node_feat
        y = data.y = torch.tensor(label)
        data.num_features = data.x.shape[-1]
        data.edge_index = edge_index
        data.num_nodes = num_nodes
        
        if args.undirected is True:
            data.edge_index = to_undirected(data.edge_index, num_nodes = data.num_nodes)

        num_classes = dataset.num_classes
        splits_idx = np.load(f'{dataset_dir}/{dataset_name}_split.npz')
        split_idx = {'train': torch.from_numpy(splits_idx['train']), 'valid': torch.from_numpy(splits_idx['valid']), 'test': torch.from_numpy(splits_idx['test'])}
    elif dataset_name =="physics":
        dataset_dir = f'{data_root}/NAGformer_small/'
        dataset = Coauthor(dataset_dir, name='Physics')
        data_o = dataset[0]
          
        edge_index = data_o.edge_index
        node_feat = data_o.x
        label = data_o.y
        num_nodes = data_o.num_nodes

        class MyObject:
            pass
        data = MyObject()
        x = data.x = node_feat
        y = data.y = torch.tensor(label)
        data.num_features = data.x.shape[-1]
        data.edge_index = edge_index
        data.num_nodes = num_nodes

        if args.undirected is True:
            data.edge_index = to_undirected(data.edge_index, num_nodes = data.num_nodes)

        num_classes = dataset.num_classes
        splits_idx = np.load(f'{dataset_dir}/{dataset_name}_split.npz')
        split_idx = {'train': torch.from_numpy(splits_idx['train']), 'valid': torch.from_numpy(splits_idx['valid']), 'test': torch.from_numpy(splits_idx['test'])}
    elif dataset_name =="wikics":
        dataset_dir = f'{data_root}/NAGformer_small/WikiCS/'
        dataset = WikiCS(dataset_dir)
        data_o = dataset[0]
          
        edge_index = data_o.edge_index
        node_feat = data_o.x
        label = data_o.y
        num_nodes = data_o.num_nodes

        class MyObject:
            pass
        data = MyObject()
        x = data.x = node_feat
        y = data.y = torch.tensor(label)
        data.num_features = data.x.shape[-1]
        data.edge_index = edge_index
        data.num_nodes = num_nodes

        #################################################################
        
        if args.undirected is True:
            data.edge_index = to_undirected(data.edge_index, num_nodes = data.num_nodes)

        num_classes = dataset.num_classes
        train_idx = (data_o.train_mask[:,args.splits_idx] == True).nonzero().squeeze()
        valid_idx = (data_o.val_mask[:,args.splits_idx] == True).nonzero().squeeze()
        test_idx = (data_o.test_mask == True).nonzero().squeeze()
        split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    elif dataset_name =="deezer":
        deezer = scipy.io.loadmat(f'{data_root}/deezer/deezer-europe.mat')

        A, label, features = deezer['A'], deezer['label'], deezer['features']
        edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
        node_feat = torch.tensor(features.todense(), dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long).squeeze()
        num_nodes = label.shape[0]

        class MyObject:
            pass
        data = MyObject()
        x = data.x = node_feat
        y = data.y = torch.tensor(label)
        data.num_features = data.x.shape[-1]
        data.edge_index = edge_index
        data.num_nodes = num_nodes

        if args.undirected is True:
            data.edge_index = to_undirected(data.edge_index, num_nodes = data.num_nodes)

        num_classes = label.max().item() + 1
        
        split_file = f'{data_root}/deezer/splits/deezer-europe_split_{args.seed}.pt'
        if not os.path.exists(split_file):
            train_idx, valid_idx, test_idx = rand_train_test_idx(y, train_prop=0.5)
            split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
            torch.save(split_idx, split_file)
        else:
          split_idx = torch.load(split_file)
    elif dataset_name in ('film', 'cornell', 'actor', 'texas', 'wisconsin'):
        data, num_classes, split_idx, x, y = load_geom_gcn_dataset(args, data_root, dataset_name)
    elif dataset_name in ('cora?', 'citeseer?', 'pubmed?'):
      transform = T.NormalizeFeatures()
      torch_dataset = Planetoid(root=data_root,
                              name=dataset_name, transform=transform)
      data_o = torch_dataset[0]

      edge_index = data_o.edge_index
      node_feat = data_o.x
      label = data_o.y
      num_nodes = data_o.num_nodes

      class MyObject:
          pass
      data = MyObject()
      x = data.x = node_feat
      y = data.y = torch.tensor(label)
      data.num_features = data.x.shape[-1]
      data.edge_index = edge_index
      data.num_nodes = num_nodes

      #################################################################
      data.edge_index = to_undirected(data.edge_index, num_nodes = data.num_nodes)

      num_classes = torch_dataset.num_classes
      
      train_idx = (torch_dataset.data.train_mask == True).nonzero().squeeze()
      valid_idx = (torch_dataset.data.val_mask == True).nonzero().squeeze()
      test_idx = (torch_dataset.data.test_mask == True).nonzero().squeeze()
      split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    elif dataset_name in {"aminer", "reddit", "Amazon2M"}:
        adj, features, labels, idx_train, idx_val, idx_test, idx_unlabel = load_data_NA_large(args, data_root, dataset_name)

        coo_matrix = adj.tocoo()
        row_indices = coo_matrix.row
        col_indices = coo_matrix.col

        adj = torch.tensor([row_indices, col_indices], dtype=torch.long)
        
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels)
        idx_train = torch.tensor(idx_train)
        idx_val = torch.tensor(idx_val)
        idx_test = torch.tensor(idx_test)

        labels = torch.argmax(labels, -1).squeeze(0)
          
        num_nodes = features.shape[0]
          
        class MyObject:
            pass
        data = MyObject()
        x = data.x = features
        y = data.y = torch.tensor(labels)
        data.num_features = data.x.shape[-1]
        data.edge_index = adj
        data.num_nodes = num_nodes

        #################################################################
        if args.undirected:
            data.edge_index = to_undirected(data.edge_index, num_nodes = data.num_nodes)
        num_classes = labels.max().item() + 1
        
        split_idx = {'train': idx_train, 'valid': idx_val, 'test': idx_test}
    elif dataset_name in {'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions', 'chameleon', 'squirrel'}:
        edge_index, node_features, labels, idx_train, idx_val, idx_test = load_data_hetero_graph_small(args, data_root, dataset_name)
        num_nodes = node_features.shape[0]
        class MyObject:
            pass
        data = MyObject()
        x = data.x = node_features
        y = data.y = torch.tensor(labels)
        data.num_features = data.x.shape[-1]
        data.edge_index = edge_index
        data.num_nodes = num_nodes

        num_classes = labels.max().item() + 1
        
        split_idx = {'train': idx_train[args.splits_idx], 'valid': idx_val[args.splits_idx], 'test': idx_test[args.splits_idx]}
        
    return data, num_classes, split_idx, x, y

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])

def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def col_normalize(mx):
    """Column-normalize sparse matrix"""
    scaler = StandardScaler()
    mx = scaler.fit_transform(mx)
    return mx
  
def load_data_NA_large(args, data_dir, dataset_str, split_seed=0, renormalize=False):
    """Load data."""
    path = f'{data_dir}/NAGformer_large/{dataset_str}/'
    
    if dataset_str == 'aminer':
        
        adj = pkl.load(open(os.path.join(path, "{}.adj.sp.pkl".format(dataset_str)), "rb"))
        features = pkl.load(
            open(os.path.join(path, "{}.features.pkl".format(dataset_str)), "rb"))
        labels = pkl.load(
            open(os.path.join(path, "{}.labels.pkl".format(dataset_str)), "rb"))
        random_state = np.random.RandomState(split_seed)
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
        idx_unlabel = np.concatenate((idx_val, idx_test))
        features = col_normalize(features)
    
    elif dataset_str in ['reddit']:
        adj = sp.load_npz(os.path.join(path, '{}_adj.npz'.format(dataset_str)))
        features = np.load(os.path.join(path, '{}_feat.npy'.format(dataset_str)))
        labels = np.load(os.path.join(path, '{}_labels.npy'.format(dataset_str))) 
        # print(labels.shape, list(np.sum(labels, axis=0)))
        random_state = np.random.RandomState(split_seed)
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)    
        idx_unlabel = np.concatenate((idx_val, idx_test))
        # print(dataset_str, features.shape)
    
    elif dataset_str in ['Amazon2M']:
        adj = sp.load_npz(os.path.join(path, '{}_adj.npz'.format(dataset_str)))
        features = np.load(os.path.join(path, '{}_feat.npy'.format(dataset_str)))
        labels = np.load(os.path.join(path, '{}_labels.npy'.format(dataset_str)))
        # print(labels.shape, list(np.sum(labels, axis=0)))
        random_state = np.random.RandomState(split_seed)
        class_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_size=20* class_num, val_size=30 * class_num)
        idx_unlabel = np.concatenate((idx_val, idx_test))
    
    else:
        raise NotImplementedError

    if renormalize:
        adj = adj + sp.eye(adj.shape[0])
        D1 = np.array(adj.sum(axis=1))**(-0.5)
        D2 = np.array(adj.sum(axis=0))**(-0.5)
        D1 = sp.diags(D1[:, 0], format='csr')
        D2 = sp.diags(D2[0, :], format='csr')

        A = adj.dot(D1)
        A = D2.dot(A)
        adj = A

    split_file = f'{data_dir}/NAGformer_large/splits/{dataset_str}_split_{args.seed}.pt'
    if not os.path.exists(split_file):
        split_idx = {'train': idx_train, 'valid': idx_val, 'test': idx_test}
        torch.save(split_idx, split_file)
    else:
        split_idx = torch.load(split_file)
        idx_train = split_idx['train']
        idx_val = split_idx['valid']
        idx_test = split_idx['test']
        
    return adj, features, labels, idx_train, idx_val, idx_test, idx_unlabel

def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding
    
    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()
    
    return label, features

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx

def even_quantile_labels(vals, nclasses=5, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    return label

def load_geom_gcn_dataset(args, data_dir, name):
    splits_list_file_path = f'{data_dir}/geom-gcn/splits/'
    graph_adjacency_list_file_path = f'{data_dir}/geom-gcn/{name}/out1_graph_edges.txt'
    graph_node_features_and_labels_file_path = f'{data_dir}/geom-gcn/{name}/out1_node_feature_label.txt'

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.tocoo().astype(np.float32)
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    def preprocess_features(feat):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(feat.sum(1))
        rowsum = (rowsum == 0) * 1 + rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        feat = r_mat_inv.dot(feat)
        return feat

    features = preprocess_features(features)

    edge_index = torch.from_numpy(
        np.vstack((adj.row, adj.col)).astype(np.int64))
    node_feat = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    num_nodes = node_feat.shape[0]

    class MyObject:
        pass
    data = MyObject()
    x = data.x = node_feat
    y = data.y = torch.tensor(labels)
    data.num_features = data.x.shape[-1]
    data.edge_index = edge_index
    data.num_nodes = num_nodes

    #################################################################
    if args.undirected:
        data.edge_index = to_undirected(data.edge_index, num_nodes = data.num_nodes)

    num_classes = labels.max().item() + 1
    split_file = f'{splits_list_file_path}/{name}_split_{args.seed}.pt'
    if not os.path.exists(split_file):
        train_idx, valid_idx, test_idx = rand_train_test_idx(y, train_prop=0.5)
        split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        torch.save(split_idx, split_file)
    else:
      split_idx = torch.load(split_file)

    return data, num_classes, split_idx, x, y


def load_data_hetero_graph_small(args, data_dir, name):
    path = f'{data_dir}/hetero-graphs/'

    if name in ('chameleon', 'squirrel'):
        name = f'{name}_filtered'
    
    data = np.load(os.path.join(path, f'{name.replace("-", "_")}.npz'))
    
    node_features = torch.tensor(data['node_features'])
    labels = torch.tensor(data['node_labels'])
    edges = torch.tensor(data['edges']).t()
    if args.undirected is True:
        edges = to_undirected(edge_index=edges, num_nodes=node_features.shape[0])

    train_masks = torch.tensor(data['train_masks'])
    val_masks = torch.tensor(data['val_masks'])
    test_masks = torch.tensor(data['test_masks'])

    train_idx_list = [torch.where(train_mask)[0] for train_mask in train_masks]
    val_idx_list = [torch.where(val_mask)[0] for val_mask in val_masks]
    test_idx_list = [torch.where(test_mask)[0] for test_mask in test_masks]

    return edges, node_features, labels, train_idx_list, val_idx_list, test_idx_list
  
def load_twitch_gamer_dataset(args, data_dir, task="mature", normalize=True):
    linkx_data_root = f'{data_dir}/linkx/'
    if not os.path.exists(f'{linkx_data_root}/twitch-gamer_feat.csv'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['twitch-gamer_feat'],
            dest_path=f'{linkx_data_root}/twitch-gamer_feat.csv',
            showsize=True
        )
    if not os.path.exists(f'{linkx_data_root}/twitch-gamer_edges.csv'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['twitch-gamer_edges'],
            dest_path=f'{linkx_data_root}/twitch-gamer_edges.csv',
            showsize=True
        )

    edges = pd.read_csv(f'{linkx_data_root}/twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{linkx_data_root}/twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)

    label, features = load_twitch_gamer(nodes, "mature")
    node_feat = torch.tensor(features, dtype=torch.float)
    node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
    node_feat = node_feat / node_feat.std(dim=0, keepdim=True)

    class MyObject:
        pass
    data = MyObject()
    x = data.x = node_feat
    y = data.y = torch.tensor(label)
    data.num_features = data.x.shape[-1]
    data.edge_index = edge_index
    data.num_nodes = num_nodes

    #################################################################
    if args.undirected is True:
        data.edge_index = to_undirected(data.edge_index, num_nodes = data.num_nodes)

    num_classes = 2
    split_file = f'{data_dir}/linkx/splits/twitch_gamer_split_{args.seed}.pt'
    if not os.path.exists(split_file):
        train_idx, valid_idx, test_idx = rand_train_test_idx(y, train_prop=0.5)
        split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        torch.save(split_idx, split_file)
    else:
      split_idx = torch.load(split_file)
    
    return data, num_classes, split_idx, x, y
  
def load_snap_patents_mat(args, data_dir, num_classes=5):
    linkx_data_root = f'{data_dir}/linkx/'
    fulldata = scipy.io.loadmat(f'{linkx_data_root}/snap_patents.mat')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)

    num_nodes = int(fulldata['num_nodes'])
    node_feat = torch.tensor(fulldata['node_feat'].todense(), dtype=torch.float)

    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, num_classes, verbose=False)
    label = torch.tensor(label, dtype=torch.long)

    class MyObject:
        pass
    data = MyObject()
    x = data.x = node_feat
    y = data.y = label
    data.num_features = data.x.shape[-1]

    data.edge_index = edge_index
    data.num_nodes = num_nodes

    if args.undirected is True:
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)


    split_file = f'{data_dir}/linkx/splits/snap_patents_split_{args.seed}.pt'
    if not os.path.exists(split_file):
        train_idx, valid_idx, test_idx = rand_train_test_idx(y, train_prop=0.5)
        split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        torch.save(split_idx, split_file)
    else:
      split_idx = torch.load(split_file)
    return data, num_classes, split_idx, x, y

def load_genius(args, data_dir):
    linkx_data_root = f'{data_dir}/linkx/'

    fulldata = scipy.io.loadmat(f'{linkx_data_root}/genius.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
    label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    class MyObject:
        pass
    data = MyObject()
    x = data.x = node_feat
    y = data.y = label
    data.num_features = data.x.shape[-1]
    data.edge_index = edge_index
    data.num_nodes = num_nodes

    if args.undirected is True:
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    num_classes = 2
    split_file = f'{data_dir}/linkx/splits/genius_split_{args.seed}.pt'
    if not os.path.exists(split_file):
        train_idx, valid_idx, test_idx = rand_train_test_idx(y, train_prop=0.5)
        split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        torch.save(split_idx, split_file)
    else:
      split_idx = torch.load(split_file)
    return data, num_classes, split_idx, x, y
  
def load_pokec_mat(args, data_dir):
    linkx_data_root = f'{data_dir}/linkx/'
    if not os.path.exists(f'{linkx_data_root}/pokec.mat'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['pokec'],
            dest_path=f'{linkx_data_root}/pokec.mat',
            showsize=True
        )

    fulldata = scipy.io.loadmat(f'{linkx_data_root}/pokec.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])

    class MyObject:
        pass
    data = MyObject()
    x = data.x = node_feat
    y = data.y = torch.tensor(fulldata['label'].flatten(), dtype=torch.long)
    data.num_features = data.x.shape[-1]
    data.edge_index = edge_index
    data.num_nodes = num_nodes

    if args.undirected is True:
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    num_classes = 2

    split_file = f'{data_dir}/linkx/splits/pokec_split_{args.seed}.pt'
    if not os.path.exists(split_file):
        train_idx, valid_idx, test_idx = rand_train_test_idx(y, train_prop=0.5)
        split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        torch.save(split_idx, split_file)
    else:
      split_idx = torch.load(split_file)
      
    return data, num_classes, split_idx, x, y

def load_arxiv_year_dataset(args, data_dir, num_classes=5):
    data_root = f'{data_dir}/ogb/'
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=data_root)
    data = dataset[0]
    
    x = data.x

    label = even_quantile_labels(data.node_year.numpy().flatten(), nclasses=num_classes, verbose=False)
    y = torch.as_tensor(label)
    split_file = f'{data_dir}/linkx/splits/arxiv_year_split_{args.seed}.pt'
    if not os.path.exists(split_file):
        train_idx, valid_idx, test_idx = rand_train_test_idx(y, train_prop=0.5)
        split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        torch.save(split_idx, split_file)
    else:
      split_idx = torch.load(split_file)
      
    if args.undirected is True:
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask
  
    return data, num_classes, split_idx, x, y
  
def load_ogbn_dataset(args, data_dir, dataset_name):
    data_root = f'{data_dir}/ogb/'
    dataset = PygNodePropPredDataset(name=dataset_name, root=data_root)
    num_classes = dataset.num_classes
    data = dataset[0]
    if args.undirected is True:
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    split_idx = dataset.get_idx_split()
    x = data.x
    y = data.y.squeeze()

    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask

    return data, num_classes, split_idx, x, y
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  