import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from model import LGMformer
from data import get_dataset
from Config import get_params
from lr import PolynomialDecayLR
from sampler import Subgraph_Sampler, Subgraph_Sampler_Nei

import sys
sys.path.append('.')
sys.path.append("..")


def train(args, model, loader, criterion, eval_func, x, pos_enc, y, optimizer, lr_scheduler, device, num_classes, batch_size, spilt_idx, epoch, spilt_type, conv_type):
    model.train()

    total_loss = 0
    num_nodes = x.shape[0]
    full_out = torch.zeros(num_nodes, num_classes, device=device)
    for edge_index, node_idx, batch_size in loader:
        features = x[node_idx]
        labels = y[node_idx]
        
        features = features.to(device)
        labels = labels.to(device)
        edge_index = edge_index.to(device)
        
        optimizer.zero_grad()
        out, loss_a = model(features, edge_index, batch_size, conv_type)
        out = out[:batch_size]
        labels = labels[:batch_size]
        o_labels = labels
        if args.is_acc is False:
            labels = F.one_hot(labels, num_classes=num_classes).float() if labels.dim()==1 else labels
        loss = criterion(out, labels)
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        full_out[node_idx[:batch_size]] = out
    acc = eval_func(y[spilt_idx], full_out[spilt_idx])
    return total_loss, acc

@torch.no_grad()
def test(args, model, loader, criterion, eval_func, x, pos_enc, y, device, num_classes, batch_size, spilt_idx, epoch, spilt_type, conv_type):
    model.eval()

    total_loss = 0
    num_nodes = x.shape[0]
    full_out = torch.zeros(num_nodes, num_classes, device=device)
    for edge_index, node_idx, batch_size in loader:
        features = x[node_idx]
        labels = y[node_idx]

        features = features.to(device)
        labels = labels.to(device)
        edge_index = edge_index.to(device)
        
        out, _ = model(features, edge_index, batch_size, conv_type)
        
        out = out[:batch_size]
        labels = labels[:batch_size]
        if args.is_acc is False:
            labels = F.one_hot(labels, num_classes=num_classes).float() if labels.dim()==1 else labels
        loss = criterion(out, labels)

        total_loss += loss.item()
        
        full_out[node_idx[:batch_size]] = out
    acc = eval_func(y[spilt_idx], full_out[spilt_idx])

    return total_loss, acc


def main():
    args = get_params()
    print(args)
    device = args.device
    sizes_str= ''
    for size_i in args.sizes:
      sizes_str = f'{sizes_str}-{size_i}'
    if args.dataset in {'cora', 'citeseer', 'computer', 'photo', 'pubmed', 'cs', 'physics',  'wikics', 'deezer', 'film', 'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions', 'chameleon', 'squirrel'}:  
        loss_dict_save_name = f'{args.part_name}_{args.dataset}_{args.seed}_{args.splits_idx}_{args.pos_enc_type}_{args.feature_hops}_{args.hidden_dim}_{args.batch_size}_{args.num_heads}_{args.peak_lr}_{args.weight_decay}_{args.end_lr}_{args.attn_dropout}_{args.ff_dropout}_{args.epochs}_{args.max_patience}_{args.compare_valid_acc}_{args.need_sample}_{sizes_str}_{args.undirected}_{args.test_start_epoch}_{args.warmup_epochs}_{args.num_layers}_{args.conv_type}_{args.token_type}_ab+-4tokens'
        loss_dict_save_name = f'{loss_dict_save_name}_centers{args.num_centroids}'
    else:
        loss_dict_save_name = f'{args.part_name}_{args.dataset}_{args.seed}_{args.splits_idx}_{args.pos_enc_type}_{args.feature_hops}_{args.hidden_dim}_{args.batch_size}_{args.num_heads}_{args.peak_lr}_{args.weight_decay}_{args.end_lr}_{args.attn_dropout}_{args.ff_dropout}_{args.epochs}_{args.max_patience}_{args.compare_valid_acc}_{args.need_sample}_{sizes_str}_{args.undirected}_{args.test_start_epoch}_{args.warmup_epochs}_{args.num_layers}_{args.conv_type}_{args.token_type}'
        loss_dict_save_name = f'{loss_dict_save_name}_centers{args.num_centroids}'

    if os.path.exists(f'{args.save_acc_loss_dict}/{loss_dict_save_name}.pt'):
        print('exist:', f'{args.save_acc_loss_dict}/{loss_dict_save_name}.pt')
        return
    print(loss_dict_save_name)
    print('get_dataset')
    data, num_classes, split_idx, features, labels= get_dataset(args, args.dataset, args.data_root, args.hetero_train_prop)
    edge_index = data.edge_index
    
    if args.dataset == 'arxiv-year':
        dataset_name_input = 'ogbn-arxiv'
    else:
        dataset_name_input = args.dataset
        
    pos_enc_node2vec = features
    
    if args.pos_enc_type == 'node2vec':
        pos_enc = torch.load(f'{args.pos_enc_path}/{dataset_name_input}_embedding_{args.global_dim}.pt', map_location='cpu')
        pos_enc = pos_enc[:,0:args.pe_dim]
        features = torch.cat((features, pos_enc), dim=1)
    elif args.pos_enc_type == 'lap':
        pos_enc = utils.laplacian_positional_encoding(edge_index, args.pe_dim, data.num_nodes) 
        features = torch.cat((features, pos_enc), dim=1)
    elif args.pos_enc_type == 'none':
        pos_enc = pos_enc_node2vec 
    elif args.pos_enc_type == 'rand':
        pos_enc = torch.rand(features.shape[0], args.pe_dim)
        features = torch.cat((features, pos_enc), dim=1)
    
    with torch.no_grad():
      print('re_features')
      re_features_name = f'{dataset_name_input}_feature+{args.pos_enc_type}_{args.feature_hops}hop'
      if dataset_name_input in {'ogbn-products'}:
          features = utils.re_features_batch(features, edge_index, args.feature_hops, device, args.node_features_save_path, re_features_name, labels, split_idx)
      else:
          features = utils.re_features(features, edge_index, args.feature_hops, device, args.node_features_save_path, re_features_name, labels, split_idx)
      features = features.squeeze()
      args.feature_hops = features.shape[1] - 1
   
    
    model = LGMformer(
      args=args,
      num_nodes=data.num_nodes,
      in_channels=features.size(-1),
      out_channels=num_classes,
      device=device
    ).to(device)

    # print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))
    if args.need_sample:
        train_loader = Subgraph_Sampler_Nei(edge_index, node_idx=split_idx['train'],
                                    num_nodes=data.num_nodes, sizes=args.sizes,
                                    batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers)
        valid_loader = Subgraph_Sampler_Nei(edge_index, node_idx=split_idx['valid'],
                                    num_nodes=data.num_nodes, sizes=args.test_sizes,
                                    batch_size=args.test_batch_size, shuffle=True,
                                    num_workers=args.num_workers)
        test_loader = Subgraph_Sampler_Nei(edge_index, node_idx=split_idx['test'],
                                    num_nodes=data.num_nodes, sizes=args.test_sizes,
                                    batch_size=args.test_batch_size, shuffle=True,
                                    num_workers=args.num_workers)
    else:
        train_loader = Subgraph_Sampler(edge_index, node_idx=split_idx['train'],
                                    num_nodes=data.num_nodes, sizes=args.sizes,
                                    batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers)
        valid_loader = Subgraph_Sampler(edge_index, node_idx=split_idx['valid'],
                                    num_nodes=data.num_nodes, sizes=args.test_sizes,
                                    batch_size=args.test_batch_size, shuffle=True,
                                    num_workers=args.num_workers)
        test_loader = Subgraph_Sampler(edge_index, node_idx=split_idx['test'],
                                    num_nodes=data.num_nodes, sizes=args.test_sizes,
                                    batch_size=args.test_batch_size, shuffle=True,
                                    num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    if args.dataset in {'aminer', 'reddit', 'Amazon2M', 'arxiv-year', 'pokec', 'genius', 'twitch-gamer', 'ogbn-arxiv', 'ogbn-products'}:
        lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup=args.warmup_epochs,
            tot=args.epochs,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0)
    else:
        lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup=args.warmup_epochs,
            tot=args.epochs,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0)

    if args.dataset in {'minesweeper', 'tolokers', 'questions', 'genius'}:
        args.is_acc = False
        criterion = F.binary_cross_entropy_with_logits
        eval_func = utils.eval_rocauc
    elif args.dataset in {'reddit', 'aminer', 'Amazon2M'}:
        criterion = F.nll_loss
        eval_func = utils.eval_acc
    else:
        criterion = nn.CrossEntropyLoss()
        eval_func = utils.eval_acc
        
    print('start epoch')
    tarin_accs = []
    train_losses = []
    train_times = []
    train_mems = []
    
    test_accs = []
    test_losses = []
    test_times = []
    test_mems = []
    
    val_accs = []
    val_losses = []
    val_times = []
    val_mems = []
    
    test_acc_final = 0
    test_loss_final = 10000000000000
    valid_acc_final = 0
    valid_loss_final = 10000000000000
    test_acc_highest = 0
    patience = 0
    for epoch in range(1, 1 + args.epochs):
        start = time.time()
        train_loss, train_acc = train(args, model, train_loader, criterion, eval_func, features, pos_enc, labels, optimizer, lr_scheduler, device, num_classes, args.batch_size, split_idx['train'], epoch, 'train', args.conv_type)
        train_time = time.time() - start
        train_mem = torch.cuda.max_memory_allocated(device=device)
        
        print(f'Epoch: {epoch}, Train loss:{train_loss:.4f}, Train acc:{100*train_acc:.2f}')
        tarin_accs.append(train_acc)
        train_losses.append(train_loss)
        train_times.append(train_time)
        train_mems.append(train_mem)
        
        if epoch % args.test_freq == 0 and epoch > args.test_start_epoch:
            start = time.time()
            valid_loss, valid_acc = test(args, model, valid_loader, criterion, eval_func, features, pos_enc, labels, device, num_classes, args.batch_size, split_idx['valid'], epoch, 'valid', args.conv_type)
            valid_time = time.time() - start
            valid_mem = torch.cuda.max_memory_allocated(device=device)
            
            print(f'Valid loss:{valid_loss:.4f}, Valid acc: {100 * valid_acc:.2f}')
            val_accs.append(valid_acc)
            val_losses.append(valid_loss)
            val_times.append(valid_time)
            val_mems.append(valid_mem)
            
            start = time.time()
            test_loss, test_acc = test(args, model, test_loader, criterion, eval_func, features, pos_enc, labels, device, num_classes, args.batch_size, split_idx['test'], epoch, 'test', args.conv_type)
            test_time = time.time() - start
            test_mem = torch.cuda.max_memory_allocated(device=device)
            
            print(f'Test loss:{test_loss:.4f}, Test acc: {100 * test_acc:.2f}')
            test_accs.append(test_acc)
            test_losses.append(test_loss)
            test_times.append(test_time)
            test_mems.append(test_mem)
            if args.compare_valid_acc:
                if valid_acc > valid_acc_final:
                    valid_loss_final = valid_loss
                    valid_acc_final = valid_acc
                    test_acc_final = test_acc
                    patience = 0
                else:
                    patience = patience + 1
                    if patience >= args.max_patience:
                        break
            else:
                if valid_loss < valid_loss_final:
                    valid_loss_final = valid_loss
                    valid_acc_final = valid_acc
                    test_acc_final = test_acc
                    patience = 0
                else:
                    patience = patience + 1
                    if patience >= args.max_patience:
                        break
                  
            if test_acc > test_acc_highest:
                test_acc_highest = test_acc
    
    print('test_acc_final: ', test_acc_final)
    print('test_acc_highest: ', test_acc_highest)
    
    save_dict={
      'parames': str(args.as_dict()),
      'test_acc_final': test_acc_final,
      'test_acc_highest': test_acc_highest,
      'tarin_accs': tarin_accs,
      'train_losses': train_losses,
      'train_times': train_times,
      'train_mems': train_mems,
      'test_accs': test_accs,
      'test_losses': test_losses,
      'test_times': test_times,
      'test_mems': test_mems,
      'val_accs': val_accs,
      'val_losses': val_losses,
      'val_times': val_times,
      'val_mems': val_mems,
    }
    
    torch.save(save_dict, f'{args.save_acc_loss_dict}/{loss_dict_save_name}.pt')
    
    load_save_dict = torch.load(f'{args.save_acc_loss_dict}/{loss_dict_save_name}.pt')
    print(load_save_dict)
    
    
    
    
if __name__ == "__main__":
    main()
