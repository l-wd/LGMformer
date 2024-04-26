import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import NeighborSampler
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from sklearn.metrics import roc_auc_score

class MessageProp(MessagePassing):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_self_loops = True

    def forward(self, x, edge_index, edge_weight = None):

        edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight, x.size(self.node_dim), self.add_self_loops, dtype=x.dtype)

        edge_weight.requires_grad_(True)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
      
def re_features(features, edge_index, K, device, save_path, re_features_name, labels, split_idx):
    if not os.path.exists(f'{save_path}/{re_features_name}.pt'):
        nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])
        ms = MessageProp()
        
        for i in range(features.shape[0]):
            nodes_features[i, 0, 0, :] = features[i]

        x = features + torch.zeros_like(features)
        print('start re_features K hops')
        for i in range(K):
            x = ms(x, edge_index)
            for index in range(features.shape[0]):
                nodes_features[index, 0, i + 1, :] = x[index]        
        nodes_features = nodes_features.squeeze()

        torch.save(nodes_features.cpu(), f'{save_path}/{re_features_name}.pt')
    else:
        nodes_features = torch.load(f'{save_path}/{re_features_name}.pt')
    return nodes_features

def re_features_batch(features, edge_index, K, device, save_path, re_features_name, labels, split_idx):
    if not os.path.exists(f'{save_path}/{re_features_name}_batch.pt'):
        features = features.to(device)
        edge_index = edge_index.to(device)
        nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])
        ms = MessageProp().to(device)
        
        subgraph_loader = NeighborSampler(edge_index, node_idx=None, sizes=[-1],
                                      batch_size=4096, shuffle=False,
                                      num_workers=12)
        
        for i in range(features.shape[0]):

            nodes_features[i, 0, 0, :] = features[i]

        x = features + torch.zeros_like(features)
        print('start re_features_batch K hops')
        for i in range(K):
            x_all = x
            x_all_new = torch.empty(features.shape[0], features.shape[1])
            for batch_size, n_id, adj in subgraph_loader:
                batch_nodes = n_id[:batch_size]
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x = ms(x, edge_index)
                
                x_all_new[batch_nodes] = x[:batch_size].cpu()
            x = x_all_new
            for index in range(features.shape[0]):
                nodes_features[index, 0, i + 1, :] = x[index]        

        nodes_features = nodes_features.squeeze()
        torch.save(nodes_features.cpu(), f'{save_path}/{re_features_name}_batch.pt')
    else:
        nodes_features = torch.load(f'{save_path}/{re_features_name}_batch.pt')
    return nodes_features

def eval_acc(y_true, y_pred):
    if len(y_true.shape) == 1:
        y_true = y_true.unsqueeze(1)
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


def eval_rocauc(y_true, y_pred):
    if len(y_true.shape) == 1:
        y_true = y_true.unsqueeze(1)
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            rocauc_list.append(score)
    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)


def laplacian_positional_encoding(edge_index, pos_enc_dim, num_nodes):
    import torch
    from scipy.sparse.linalg import eigsh, eigs
    from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops, get_laplacian
    laplacian_, lap_weight = get_laplacian(edge_index, normalization="sym", num_nodes=num_nodes)
    # get laplacian (sparse matrix format)
    laplacian = to_scipy_sparse_matrix(laplacian_, lap_weight)
    # calculate largest eigen value
    EigVal, EigVec = eigs(laplacian, k=pos_enc_dim+1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
  
    return lap_pos_enc
