import torch
import psutil
import os
import sys
import time
import argparse
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import seaborn as sns
import networkx as nx
import pandas as pd
import datetime
from torch.optim import Adam
from sklearn import metrics
from networkx.algorithms import tree
from scipy.sparse import csr_matrix
from torch_geometric.utils import to_undirected, negative_sampling
from torch.utils.data import DataLoader
from torch_geometric_signed_directed.utils import in_out_degree
####
from data_loader import dataloader
from model.dhypr import LPModel
from dhypr_utils import get_processed_adj
from split import link_split
from utils import get_metric_score, Logger, get_logger, get_config_dir, init_seed

log_print = get_logger('testrun', 'log', get_config_dir())


def train(model, feat, optimizer, lr_scheduler, adj, train_pos, train_neg, batch_size):
    model.train()
    total_loss, total_samples = 0, 0
    device = feat.device

    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        #num_nodes = feat.size(0)

        embeddings_all = model.encode(feat, adj)
        embeddings = embeddings_all[-1]

        pos_edge = train_pos[perm].to(device)
        if train_neg is None:
            #do some random sampling on every batch.
            neg_edge = negative_sampling(train_pos.t(), num_neg_samples=pos_edge.size(0), force_undirected=False)
            neg_edge = neg_edge.t().to(device)
        else:
            #using the fixed splitting
            neg_edge = train_neg[perm].to(device)

        loss = model.get_loss(args, embeddings, pos_edge, neg_edge)
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        num_samples = pos_edge.size(1)
        total_loss += loss.item() * num_samples
        total_samples += num_samples

    return total_loss / total_samples

@torch.no_grad()
def test_edge(model, input_data, emb, batch_size, device):
    preds = []
    for perm  in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].to(device)
        pred, _ = model.decode(emb, edge)
        preds += [pred.cpu()]

    #print(preds)
    pred_all = torch.cat(preds, dim=0)

    return pred_all


@torch.no_grad()
def test(model, feat, adj, data, batch_size, criterion=None):
    device = feat.device
    model.eval()
    embeddings_all = model.encode(feat, adj)
    emb1 = embeddings_all[-1]
    
    #using the fixed splitting to evaluate
    train_pos = data['train']['pos']
    train_neg = data['train']['neg']
    
    val_pos = data['val']['pos']
    val_neg = data['val']['neg']
    test_pos = data['test']['pos']
    test_neg = data['test']['neg']

    #pos_train_pred = test_edge(predictor, train_pos, emb1, batch_size, device)
    #neg_train_pred = test_edge(predictor, train_neg, emb1, batch_size, device)
    pos_valid_pred = test_edge(model, val_pos, emb1, batch_size, device)
    neg_valid_pred = test_edge(model, val_neg, emb1, batch_size, device)
    pos_test_pred = test_edge(model, test_pos, emb1, batch_size, device)
    neg_test_pred = test_edge(model, test_neg, emb1, batch_size, device)

    if criterion is not None:
        #CE loss
        #pos_train_pred, neg_train_pred = torch.exp(pos_train_pred[:,1]), torch.exp(neg_train_pred[:,1])
        pos_valid_pred, neg_valid_pred = torch.exp(pos_valid_pred[:,1]), torch.exp(neg_valid_pred[:,1])
        pos_test_pred,  neg_test_pred = torch.exp(pos_test_pred[:,1]), torch.exp(neg_test_pred[:,1])
        
    #pos_train_pred, neg_train_pred = torch.flatten(pos_train_pred), torch.flatten(neg_train_pred)
    pos_valid_pred, neg_valid_pred = torch.flatten(pos_valid_pred), torch.flatten(neg_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)

    pos_train_pred = None
    neg_train_pred = None
    result = get_metric_score(pos_train_pred, neg_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

    return result


def run(data, link_data, model, args, split, save_path):
    device = args.device
    batch_size = args.batch_size
    eval_metric = args.metric
    #embedding_type = args.embedding

    run_logs = None
    if args.save_res and split < 1:
        run_logs = os.path.join(save_path,"first_run_logs.txt")

    edge_index = link_data[split]['graph'] #The observed graph for training propagation
    if args.undirected:
        edge_index = to_undirected(edge_index, data.x.shape[0])

    if args.origin_feat:
        feat = data.x.to(device)
    else:
        #use in/out degree as feature
        feat = in_out_degree(edge_index, size=data.x.shape[0]).to(device)
        print("Use in/out degree as feature!")

    args.feat_dim = feat.shape[-1]
    train_pos = link_data[split]['train']['pos']
    if args.neg_sampler == "fixed":
        train_neg = link_data[split]['train']['neg']
    else:
        train_neg = None

    load_adj_path = os.path.join(args.adj_path, args.dataset, str(split), "adj.pt")
    #breakpoint()
    if os.path.exists(load_adj_path):
        adj = torch.load(load_adj_path)
    else:
        adj = get_processed_adj(edge_index, args.normalize_adj)
        os.makedirs(os.path.dirname(load_adj_path), exist_ok=True)
        torch.save(adj, load_adj_path)
    adj = {key: value.to(device) for key, value in adj.items()}

    #print(adj)
    #breakpoint()

    #building model
    args.n_nodes = data.x.shape[0]
    args.nb_false_edges = len(train_neg)
    args.nb_edges = len(train_pos)
    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    model = model(args).to(device)

    # Building predictor and loss function
    criterion = None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    
    best_valid = 0
    stopping_cnt = 0
    print("Run"+str(split+1)+" Start Training!")
    for epoch in range(1, args.epochs+1):
        st_time = time.time()

        loss = train(model, feat, optimizer, lr_scheduler, adj, train_pos, train_neg, batch_size)
        results = test(model, feat, adj, link_data[split], batch_size)
        
        for key, result in results.items():
            loggers[key].add_result(split, result)

        #save logs
        if run_logs is not None:
            if epoch == 1 or epoch % 5 == 0:
                with open(run_logs, 'a', encoding='utf-8') as file:
                    for key, result in results.items():
                        file.write(key+'\n')
                        train_res, valid_res, test_res = result
                        file.write(f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_res:.2f}%, '
                            f'Valid: {100 * valid_res:.2f}%, '
                            f'Test: {100 * test_res:.2f}%'+'\n')
                    file.write("=============================="+'\n')
        #print logs
        '''if epoch % args.print_steps == 0:
            for key, result in results.items():
                print(key)
                train_res, valid_res, test_res = result
                log_print.info(
                    f'Run: {split + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_res:.2f}%, '
                    f'Valid: {100 * valid_res:.2f}%, '
                    f'Test: {100 * test_res:.2f}%')
            print(time.time()-st_time)
            print('---')'''

        best_valid_current = torch.tensor(loggers[eval_metric].results[split])[:, 1].max()
        if best_valid_current > best_valid:
            best_valid = best_valid_current
            stopping_cnt = 0
        else:
            stopping_cnt += 1
            if stopping_cnt > args.early_stopping: 
                print("Early Stopping!")
                break
    #print
    '''for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(split)'''

def main(args):
    dataset = dataloader(args.dataset)
    data = dataset[0]
    save_path = None
    if args.save_res:
        ##method folder
        save_path = os.path.join(args.save_path, args.net)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ##data_time folder
        time_n = datetime.datetime.now().strftime("%m%d-%H%M%S")
        data_time = args.dataset+"-"+time_n
        save_path = os.path.join(save_path,data_time)
        os.makedirs(save_path)
        para_logs = os.path.join(save_path,"parameters.txt")
        args_dict = vars(args)
        with open(para_logs, 'w', encoding='utf-8') as file:
            file.write(f"The arguments are:\n")
            for key, value in args_dict.items():
                file.write(f"{key}: {value}\n")
    #data splitting
    link_data = link_split(
        name=args.dataset, 
        data=data, 
        num_splits=args.runs, 
        prob_test=args.prob_test, 
        prob_val=args.prob_val,  
        maintain_connect=True
    )
    #select model
    Net = LPModel

    global loggers 
    loggers= {
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs),
        'ACC': Logger(args.runs)
    }

    #running
    for i in range(args.runs):
        st_time = time.time()
        run(data, link_data, Net, args, i, save_path)
        print("Run", i, time.time()-st_time)


    #print and save
    save_results = []
    for i in range(1, args.runs+1):
        save_results.append({"Run":"Run"+str(i)})
    save_results.append({"Run":"Valid"})
    save_results.append({"Run":"Test"})
    columns_order = ["Run"]
    print("==============================")
    for key in loggers.keys():
        columns_order.append(key)
        print(key)
        all_run_result, best_valid, final_test = loggers[key].print_statistics()
        for i in range(all_run_result.shape[0]):
            save_results[i][key] = round(all_run_result[i][-1].item(), 2)
        save_results[-2][key] = best_valid
        save_results[-1][key] = final_test
    
    #save
    df = pd.DataFrame(save_results)
    df = df[columns_order]
    results_file = os.path.join(save_path,"result.xlsx")
    df.to_excel(results_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--device', type=int, default=0, help='Device index (e.g., 0 for CPU).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    # Dataset
    parser.add_argument('--dataset', type=str, default='cora_ml', help='Dataset name.')
    parser.add_argument('--prob_val', type=float, default=0.05, help='Validation set proportion.')
    parser.add_argument('--prob_test', type=float, default=0.15, help='Test set proportion.')
    parser.add_argument('--undirected', type=bool, default=False, help='Use undirected graph.')
    parser.add_argument('--origin_feat', type=bool, default=False, help='Use original features.')
    parser.add_argument('--adj_path', type=str, default="./data/dhyper", help='Use original features.')
    # Model
    parser.add_argument('--net', type=str, default='DHYPER', help='Model name.')
    parser.add_argument('--hidden', type=int, default=64, help='number of units in GNN encoder hidden layer (excpet the last output, which should be dim).')
    parser.add_argument('--feat_dim', type=int, default=2, help='Input feature dimension.')
    parser.add_argument('--dim', type=int, default=32, help='Embedding dimension.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers.')
    parser.add_argument('--embedding', type=str, default="dual", help='Embedding type.')
    parser.add_argument('--act', type=str, default='relu', help='which activation function to use')
    parser.add_argument('--c', type=str, default=None, help='hyperbolic radius, set to None for trainable curvature')
    parser.add_argument('--manifold', type=str, default='PoincareBall', help='which manifold to use')
    parser.add_argument('--model', type=str, default='DHYPR', help='which encoder to use')
    parser.add_argument('--proximity', type=int, default=2, help='largest order of proximity matrix')
    parser.add_argument('--bias', type=int, default=1, help='whether to use bias (1) or not (0)')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha for leakyrelu for attention')
    parser.add_argument('--beta', type=float, default=1.0, help='beta in gravity decoder')
    parser.add_argument('--lamb', type=float, default=0.1, help='lamda in gravity decoder')
    parser.add_argument('--use_att', type=int, default=0, help='whether to use hyperbolic attention or not')
    parser.add_argument('--n_heads', type=int, default=4, help='number of attention heads for graph attention networks, must be a divisor dim')
    parser.add_argument('--r', type=int, default=2, help='fermi-dirac decoder parameter for lp')
    parser.add_argument('--t', type=int, default=1, help='fermi-dirac decoder parameter for lp')
    # Hyperparameters
    parser.add_argument('--normalize_adj', type=int, default=1, help='whether to row-normalize the adjacency matrix')
    parser.add_argument('--dropout', type=float, default=0.05, help='DropouACt rate.')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma for lr scheduler.')
    parser.add_argument('--lr_reduce_freq', type=int, default=None, help='reduce lr every lr-reduce-freq or None to keep lr constant.')
    parser.add_argument('--wl1', type=float, default=1.0, help='loss weight term for fc regularizer')
    parser.add_argument('--wl2', type=float, default=1.0, help='loss weight term for gravity regularizer')
    # Training
    parser.add_argument('--runs', type=int, default=2, help='Number of runs.')
    parser.add_argument('--epochs', type=int, default=10, help='Max epochs.')
    parser.add_argument('--early_stopping', type=int, default=200, help='Early stopping epochs.')
    parser.add_argument('--batch_size', type=int, default=64*1024, help='Batch size.')
    parser.add_argument('--metric', type=str, default='Hits@100', help='Evaluation metric.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--momentum', type=float, default=0.999, help='momentum in optimizer.')
    parser.add_argument('--neg_sampler', type=str, default="fixed")
    # Output and Saving
    parser.add_argument('--print_steps', type=int, default=200, help='Steps between prints.')
    parser.add_argument('--save_path', type=str, default="./results", help='Path to save results.')
    parser.add_argument('--save_res', type=bool, default=False, help='Save results flag.')

    args = parser.parse_args()
    args.device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')

    print("==============================")
    print(f"The arguments are: {args}")

    init_seed(args.seed)
    main(args)
