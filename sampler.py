import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops, k_hop_subgraph
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import MessagePassing



class MessageProp(MessagePassing):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_self_loops = True

    def forward(self, x: Tensor, edge_index, edge_weight = None) -> Tensor:

        edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight, x.size(self.node_dim), self.add_self_loops, dtype=x.dtype)

        edge_weight.requires_grad_(True)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out

class Subgraph_Sampler(torch.utils.data.DataLoader):
    def __init__(self, edge_index, sizes, node_idx = None, num_nodes = None, return_e_id = True, transform = None, **kwargs):
        edge_index = edge_index.to('cpu')

        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)[0]

        self.edge_index = edge_index
        
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform

        super().__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        batch_size: int = len(batch)
        k_hop_subgraph_nodes_index, old_edge_index,  batch_nodes_index,  _ = k_hop_subgraph(batch, 1, self.edge_index, directed=True)

        node_idx = k_hop_subgraph_nodes_index
        node_idx_flag = torch.tensor([i not in batch for i in node_idx])
        node_idx = node_idx[node_idx_flag]
        node_idx = torch.cat([batch, node_idx])

        node_idx_all = torch.zeros(self.num_nodes, dtype=torch.long)
        node_idx_all[node_idx] = torch.arange(node_idx.size(0))
        new_edge_index = node_idx_all[old_edge_index]
        return new_edge_index, node_idx, batch_size

class Subgraph_Sampler_Nei(torch.utils.data.DataLoader):
    def __init__(self, edge_index, sizes, node_idx = None, num_nodes = None, return_e_id = True, transform = None, **kwargs):
        edge_index = edge_index.to('cpu')
        edge_index = add_self_loops(edge_index)[0]

        self.edge_index = edge_index
        
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform

        value = torch.arange(edge_index.size(1)) if return_e_id else None
        self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                  value=value,
                                  sparse_sizes=(num_nodes, num_nodes)).t()

        self.adj_t.storage.rowptr()

        super().__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        batch_size: int = len(batch)

        edge_index = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            row, col ,_ = adj_t.t().coo()
            edge_index_tmp = torch.stack([row, col], axis=0)
            edge_index.append(edge_index_tmp)
        edge_index = torch.cat(edge_index, dim=1)

        node_idx = torch.unique(edge_index[0])
        node_idx_flag = torch.tensor([i not in batch for i in node_idx])
        node_idx = node_idx[node_idx_flag]
        node_idx = torch.cat([batch, node_idx])

        node_idx_all = torch.zeros(self.num_nodes, dtype=torch.long)
        node_idx_all[node_idx] = torch.arange(node_idx.size(0))
        edge_index = node_idx_all[edge_index]
        
        return edge_index, node_idx, batch_size

