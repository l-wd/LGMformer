import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

from ema_cluster import VectorQuantizerEMA

class MessageProp(MessagePassing):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(*args, **kwargs)
        
    def forward(self, x: Tensor, edge_index, num_nodes = None,
                edge_weight = None) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        return out

    def message(self, x: Tensor, edge_weight) -> Tensor:
        return x if edge_weight is None else edge_weight.view(-1, 1) * x

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class NTIformer(nn.Module):
    def __init__(
        self, args, in_channels, out_channels, num_class, global_dim, num_nodes, num_heads, attn_dropout, ff_dropout, device, **kwargs,
    ):
        super(NTIformer, self).__init__()
        self.token_type = args.token_type
        self.feature_hops = args.feature_hops
        self.global_dim = global_dim
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.device = device
        self.num_heads = num_heads
        
        self.ms = MessageProp()
        
        self.self_attention_norm = nn.LayerNorm(in_channels)
        self.self_attention_dropout = nn.Dropout(ff_dropout)

        self.att_size = att_size = in_channels // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(in_channels, num_heads * att_size)
        self.linear_k = nn.Linear(in_channels, num_heads * att_size)
        self.linear_v = nn.Linear(in_channels, num_heads * att_size)
        self.att_dropout = nn.Dropout(attn_dropout)

        self.output_layer = nn.Linear(num_heads * att_size, in_channels)

        
    def forward(self, x_cen: Tensor, x_nei: Tensor, edge_index = None, former_type = None, batch_size = None):
        add_token = (x_nei + x_cen)
        mins_token = (x_nei - x_cen)
        multi_token = (x_nei * x_cen)
        if self.token_type=='full':
            edge_tokens = torch.stack((x_cen, x_nei, add_token, mins_token), dim=1)
        elif self.token_type=='hom':
            edge_tokens = torch.stack((x_cen, x_nei, add_token), dim=1)
        elif self.token_type=='het':
            edge_tokens = torch.stack((x_cen, x_nei, mins_token), dim=1)
        elif self.token_type=='multi':
            edge_tokens = torch.stack((x_cen, x_nei, multi_token), dim=1)
        elif self.token_type=='none':
            edge_tokens = torch.stack((x_cen, x_nei), dim=1)
          
        x = edge_tokens
        y = self.self_attention_norm(x)
        y = self.local_forward(y[:,:1], y, y)
        y = self.self_attention_dropout(y)
        
        out_hom = y[:, 0] + x_cen
        out_cen = self.ms(out_hom, edge_index)
        
        return out_cen, None, None
  
    def local_forward(self, q: Tensor, k: Tensor, v: Tensor):

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        
        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        
        x = x * self.scale
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        return x
    
class TransformerConv(nn.Module):
    def __init__(
        self, args, in_channels, out_channels, num_class, global_dim, num_nodes, num_heads, attn_dropout, ff_dropout, device, **kwargs,
    ):
        super(TransformerConv, self).__init__()
        num_centroids = args.num_centroids
        self.feature_hops = args.feature_hops
        self.num_centroids = args.num_centroids
        self.global_dim = in_channels
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_dropout = attn_dropout
        self.device = device
        self.num_heads = num_heads
        self.ff_dropout=ff_dropout,
        
        self.self_attention_norm = nn.LayerNorm(in_channels)
        
        self.self_attention_dropout = nn.Dropout(ff_dropout)

        self.ffn_norm = nn.LayerNorm(in_channels)
        self.ffn = FeedForwardNetwork(in_channels, out_channels, ff_dropout)
        self.ffn_dropout = nn.Dropout(ff_dropout)

        self.att_size = att_size = in_channels // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(in_channels, num_heads * att_size)
        self.linear_k = nn.Linear(in_channels, num_heads * att_size)
        self.linear_v = nn.Linear(in_channels, num_heads * att_size)
        self.att_dropout = nn.Dropout(attn_dropout)

        self.output_layer = nn.Linear(num_heads * att_size, in_channels)


        self.vq = VectorQuantizerEMA(
            num_centroids, 
            in_channels, 
            decay=0.99
        )
        self.lin_key_g = Linear(in_channels, num_heads * att_size)
        self.lin_query_g = Linear(in_channels, num_heads * att_size)
        self.lin_value_g = Linear(in_channels, num_heads * att_size)

        self.hopwise = nn.Parameter(torch.ones(2, dtype=torch.float))

        self._atten_=None
        
    def forward(self, x: Tensor, conv_type):
        if conv_type=='local':
            y = self.self_attention_norm(x)
            y = self.local_forward(y, y, y)
            y = self.self_attention_dropout(y)
            x = x + y

            y = self.ffn_norm(x)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            x = x + y
            out_local = x
            out = out_local
        else:
            # 1
            g_in=x
            out_gg3 = self.global_forward(g_in)
            out_global = out_gg3+g_in
            
            # 2
            y = self.self_attention_norm(x)
            y = self.local_forward(y, y, y)
            y = self.self_attention_dropout(y)
            x = x + y

            y = self.ffn_norm(x)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            x = x + y
            out_local = x
            out = out_local*self.hopwise[0] + out_global*self.hopwise[1]
        
        return out
   
    def global_forward(self, x):

        d, h = self.att_size, self.num_heads
        batch_size = x.size(0)
        
        q_x = x
        k_x = self.vq.get_kv()
        v_x = self.vq.get_kv()

        q = self.lin_query_g(q_x).view(batch_size, -1, h, d)
        k = self.lin_key_g(k_x).view(-1, h, d)
        v = self.lin_value_g(v_x).view(-1, h, d)

        q = q.transpose(0, 1).transpose(1, 2)  # [b, k, h, d] -> [k, b, h, d] -> [k, h, b, d]
        v = v.transpose(0, 1)                  # [n, h, d]    -> [h, n, d]
        k = k.transpose(0, 1).transpose(1, 2)  # [n, h, d]    -> [h, d, n]

        dots = torch.matmul(q, k) * self.scale

        attn = torch.softmax(dots, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)

        out = attn.matmul(v) # [k, h, b, d]
        out = out.transpose(1, 2).transpose(0, 1).contiguous()  # [b, k, h, d]
        out = out.view(batch_size, -1, h*d)

        # Update the centroids
        if self.training :
            distances = self.vq.update(q_x.view(-1, h*d))

        return out

    def local_forward(self, q: Tensor, k: Tensor, v: Tensor):

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        
        
        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        
        x = x * self.scale
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        return x
    
    
class LGMformer(torch.nn.Module):
    def __init__(self, args, num_nodes, in_channels, out_channels, device=None):
        super(LGMformer, self).__init__()
        num_layers=args.num_layers
        edge_former_num_layers=args.edge_former_num_layers
        hidden_channels=args.hidden_dim
        global_dim=args.global_dim
        num_heads=args.num_heads
        ff_dropout=args.ff_dropout
        attn_dropout=args.attn_dropout
        h_times = args.h_times
        
        self.hidden_channels = args.hidden_dim
        self.feature_hops = args.feature_hops
        self.seq_len = self.feature_hops+1
        
        self.device=device
        self.ms = MessageProp()
       
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        self.edge_trans_convs_hops = torch.nn.ModuleList()
        for _ in range(self.seq_len):
            self.edge_convs = torch.nn.ModuleList()
            self.trans_convs = torch.nn.ModuleList()
            for _ in range(edge_former_num_layers):
                self.trans_convs.append(
                    NTIformer(
                        args=args,
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        num_class=out_channels,
                        global_dim=global_dim,
                        num_nodes=num_nodes,
                        num_heads=num_heads,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                        device=device
                    )
                )
            self.edge_convs.append(self.trans_convs)    
            self.edge_trans_convs_hops.append(self.edge_convs)
        self.convs = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    args=args,
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    num_class=out_channels,
                    global_dim=global_dim,
                    num_nodes=num_nodes,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    device=device
                )
            )
            
        self.final_ln = nn.LayerNorm(hidden_channels*h_times)
        self.attn_layer = nn.Linear(2 * self.hidden_channels*h_times, 1)
        self.out_proj = nn.Linear(self.hidden_channels*h_times, int(self.hidden_channels/2))
        self.Linear1 = nn.Linear(int(self.hidden_channels/2), out_channels)


    def forward(self, x, edge_index, batch_size, conv_type):
        if conv_type=='global':
            output, _ = self.global_forward(x, edge_index, batch_size, conv_type)
        elif conv_type=='local':
            output, _ = self.local_forward(x, edge_index, batch_size, conv_type)
        else:
            output, _ = self.full_forward(x, edge_index, batch_size, conv_type)
            
        return output, None
      
    def forward_1(self, x, edge_index, batch_size, conv_type):
        x = self.fc_in(x)
        x_cen = x[:batch_size]
        x_list = []
        for hop_i, edge_conv in enumerate(self.edge_trans_convs_hops):
            x_hop = x[:,hop_i]
            x_edge_cen = x_hop[edge_index[1]]
            x_edge_nei = x_hop[edge_index[0]]
            trans_convs = edge_conv[0]
            hop_out = 0
            for i, _ in enumerate(trans_convs):
                out_cen, x_edge_nei, similarity = trans_convs[i](x_edge_cen, x_edge_nei, edge_index, 'hom', batch_size)
            hop_out = out_cen
            x_list.append(hop_out)
        
        new_x_edge = torch.stack(x_list, dim=1)
        
        x = new_x_edge[:batch_size]
        for i, conv in enumerate(self.convs):
            tensor = conv(x)
            x = tensor + x 
        output = self.final_ln(x)

        target = output[:,0,:].unsqueeze(1).repeat(1,self.seq_len-1,1)
        split_tensor = torch.split(output, [1, self.seq_len-1], dim=1)

        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]

        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        
        layer_atten = F.softmax(layer_atten, dim=1)
        
        neighbor_tensor = neighbor_tensor * layer_atten

        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)

        output = (node_tensor + neighbor_tensor).squeeze()
        
        output = self.Linear1(torch.relu(self.out_proj(output)))

    
        return torch.log_softmax(output, dim=1), None

    def full_forward(self, x, edge_index, batch_size, conv_type):
        x = self.fc_in(x)
        x_cen = x[:batch_size]
        x_list = []
        for hop_i, edge_conv in enumerate(self.edge_trans_convs_hops):
            x_hop = x[:,hop_i]
            x_edge_cen = x_hop[edge_index[1]]
            x_edge_nei = x_hop[edge_index[0]]
            trans_convs = edge_conv[0]
            hop_out = 0
            for i, _ in enumerate(trans_convs):
                out_cen, x_edge_nei, similarity = trans_convs[i](x_edge_cen, x_edge_nei, edge_index, 'hom', batch_size)
            hop_out = out_cen
            x_list.append(hop_out)
        
        new_x_edge = torch.stack(x_list, dim=1)
        
        x = new_x_edge[:batch_size]
        for i, conv in enumerate(self.convs):
            tensor = conv(x, conv_type)
            x = tensor + x 
        output = self.final_ln(x)

        target = output[:,0,:].unsqueeze(1).repeat(1,self.seq_len-1,1)
        split_tensor = torch.split(output, [1, self.seq_len-1], dim=1)

        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]

        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        
        layer_atten = F.softmax(layer_atten, dim=1)
        
        neighbor_tensor = neighbor_tensor * layer_atten

        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)

        output = (node_tensor + neighbor_tensor).squeeze()
        
        output = self.Linear1(torch.relu(self.out_proj(output)))

    
        return torch.log_softmax(output, dim=1), None
      
    def global_forward(self, x, edge_index, batch_size, conv_type):
        x = self.fc_in(x)
        x = x[:batch_size]
        for i, conv in enumerate(self.convs):
            tensor = conv(x, conv_type)
            x = tensor + x 
        output = self.final_ln(x)

        target = output[:,0,:].unsqueeze(1).repeat(1,self.seq_len-1,1)
        split_tensor = torch.split(output, [1, self.seq_len-1], dim=1)

        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]

        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        
        layer_atten = F.softmax(layer_atten, dim=1)
        
        neighbor_tensor = neighbor_tensor * layer_atten

        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)

        output = (node_tensor + neighbor_tensor).squeeze()
        
        output = self.Linear1(torch.relu(self.out_proj(output)))

    
        return torch.log_softmax(output, dim=1), None

    def local_forward(self, x, edge_index, batch_size, conv_type):
        x = self.fc_in(x)
        x_cen = x[:batch_size]
        x_list = []
        for hop_i, edge_conv in enumerate(self.edge_trans_convs_hops):
            x_hop = x[:,hop_i]
            x_edge_cen = x_hop[edge_index[1]]
            x_edge_nei = x_hop[edge_index[0]]
            trans_convs = edge_conv[0]
            hop_out = 0
            for i, _ in enumerate(trans_convs):
                out_cen, x_edge_nei, similarity = trans_convs[i](x_edge_cen, x_edge_nei, edge_index, 'hom', batch_size)
            hop_out = out_cen
            x_list.append(hop_out)
        
        new_x_edge = torch.stack(x_list, dim=1)
        
        x = new_x_edge[:batch_size]
        for i, conv in enumerate(self.convs):
            tensor = conv(x, conv_type)
            x = tensor + x 
        output = self.final_ln(x)

        target = output[:,0,:].unsqueeze(1).repeat(1,self.seq_len-1,1)
        split_tensor = torch.split(output, [1, self.seq_len-1], dim=1)

        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]

        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        
        layer_atten = F.softmax(layer_atten, dim=1)
        
        neighbor_tensor = neighbor_tensor * layer_atten

        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)

        output = (node_tensor + neighbor_tensor).squeeze()
        
        output = self.Linear1(torch.relu(self.out_proj(output)))
    
        return torch.log_softmax(output, dim=1), None
