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
from torch_geometric.utils import to_undirected
from torch.utils.data import DataLoader
from dgl.base import NID, EID
from torch_geometric_signed_directed.utils import in_out_degree
####
from data_loader import dataloader
from model.duplex import DUPLEX_gat
from split import link_split
from utils import get_metric_score, Logger, get_logger, get_config_dir, init_seed
from utils_duplex import fourClassLoss, load_training_data, predictor, load_testing_data
log_print = get_logger('testrun', 'log', get_config_dir())

import ipdb

def acc(pred, label):
    correct = pred.eq(label).sum().item()
    acc = correct / len(pred)
    return acc

def train(train_dataloader, model, optimizer, loss_fun, loss_weight, args):
    #torch.autograd.set_detect_anomaly(True)
    model.train()
    total_loss = 0.0
    for step, (input_nodes, pos_graph, blocks) in enumerate(train_dataloader):
        input_am = blocks[0].srcdata['h']
        input_ph = blocks[0].srcdata['h'] 

        #output_nodes = blocks[-1].dstdata[NID] # initial ids of output nodes
        am_outputs, ph_outputs = model(blocks, input_am, input_ph)

        pos_edges = torch.cat((
            pos_graph.edges()[0].unsqueeze(-1),
            pos_graph.edges()[1].unsqueeze(-1),
            pos_graph.edata['label'].unsqueeze(-1)
            ), 1)
        pos_edges[pos_edges[:,2]==0] = \
            torch.index_select(
                pos_edges[pos_edges[:,2]==0], 
                1, 
                torch.tensor([1,0,2]).to(args.device)
                )

        loss = loss_fun(pos_edges, am_outputs, ph_outputs, loss_weight, args.origin_feat)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss = total_loss+loss.item()

        #train_label = pos_edges[:,2]
        #train_pred_label, _ = predictor(pos_edges[:,:2], am_outputs.detach(), ph_outputs.detach(), args.device) # shape (N,3)
        #train_acc= acc(train_pred_label, train_label) # four-type acc
        train_acc = 0.0
    
    return total_loss, train_acc

@torch.no_grad()
def test_edge(model, blocks, edges, args):
    input_am = blocks[0].srcdata['h']
    input_ph = blocks[0].srcdata['h']
    
    output_nodes = blocks[-1].dstdata[NID]
    am_outputs, ph_outputs = model(blocks, input_am, input_ph)

    global2batch = torch.zeros(max(output_nodes)+1,dtype=int).to(args.device)
    global2batch[output_nodes]=torch.arange(0, am_outputs.shape[0],dtype=int).to(args.device) # global idx to batch idx
    batch_edges = torch.cat((global2batch[edges[:,0]].unsqueeze(1),global2batch[edges[:,1]].unsqueeze(1)),dim=1)

    _, pred = predictor(batch_edges, am_outputs, ph_outputs, args.device) 

    return pred.cpu()

@torch.no_grad()
def test(model, data, val_pos_blocks, val_neg_blocks, test_pos_blocks, test_neg_blocks, args):
    model.eval()
    val_pos = data['val']['pos'].to(args.device)
    val_neg = data['val']['neg'].to(args.device)
    test_pos = data['test']['pos'].to(args.device)
    test_neg = data['test']['neg'].to(args.device)

    pos_valid_pred = test_edge(model, val_pos_blocks, val_pos, args)
    neg_valid_pred = test_edge(model, val_neg_blocks, val_neg, args)
    pos_test_pred = test_edge(model, test_pos_blocks, test_pos, args)
    neg_test_pred = test_edge(model, test_neg_blocks, test_neg, args)

    pos_valid_pred, neg_valid_pred = torch.flatten(pos_valid_pred), torch.flatten(neg_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)

    pos_train_pred = None
    neg_train_pred = None
    result = get_metric_score(pos_train_pred, neg_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

    return result


def run(data, link_data, model, args, split, save_path):
    #split = args.split
    device = args.device
    batch_size = args.batch_size
    eval_metric = args.metric

    edge_index = link_data[split]['graph']

    run_logs = None
    if args.save_res and split < 1:
        run_logs = os.path.join(save_path,"first_run_logs.txt")
    num_nodes = data.x.shape[0]
    if args.origin_feat:
        feat = data.x

    else:
        #use in/out degree as feature
        feat = in_out_degree(edge_index, size=data.x.shape[0])
        feat = feat.float()
        print("Use in/out degree as feature!")

    args.input_dim = feat.shape[-1]
    #building model
    model = model(args).to(device)

    # Building predictor and loss function
    loss_fun = fourClassLoss(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #training data
    train_dataloader, train_graph  = load_training_data(args, link_data[split], feat, num_nodes, split)
    val_pos_blocks, val_neg_blocks, test_pos_blocks, test_neg_blocks = load_testing_data(args, link_data[split], train_graph)

    best_valid = 0
    stopping_cnt = 0
    print("Run"+str(split+1)+" Start Training!")
    loss_weight = args.loss_weight
    for epoch in range(1, args.epochs+1):
        st_time = time.time()
        loss, train_acc = train(train_dataloader, model, optimizer, loss_fun, loss_weight, args)
        loss_weight = loss_weight*(1-args.loss_decay) # connection-aware loss weight decay

        #st_time = time.time()
        results = test(model, link_data[split], val_pos_blocks, val_neg_blocks, test_pos_blocks, test_neg_blocks, args)
        #continue

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
    #for key in loggers.keys():
    #    print(key)
    #    loggers[key].print_statistics(split)

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
        time = datetime.datetime.now().strftime("%m%d-%H%M%S")
        data_time = args.dataset+"-"+time
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
    Net = DUPLEX_gat

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
        run(data, link_data, Net, args, i, save_path)

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
    # Model
    parser.add_argument('--net', type=str, default='DUPLEX', help='Model name.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer size.')
    parser.add_argument('--input_dim', type=int, default=128, help='Input feature dimension.')
    parser.add_argument('--output_dim', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--nlayers', type=int, default=3, help='Number of layers.')
    parser.add_argument('--embedding', type=str, default="dual", help='Embedding type.')
    parser.add_argument('--fusion', type=str, default="add")
    parser.add_argument('--head', type=int, default=1)
    # Hyperparameters
    parser.add_argument('--dropout', type=float, default=0.5, help='DropouACt rate.')
    parser.add_argument('--loss_weight', type=float, default=0.3, dest = 'loss_weight', help='bce loss weight')
    parser.add_argument('--loss_decay', type=float, default=1e-4, dest = 'loss_decay', help='weight decay per epoch')
    # Training
    parser.add_argument('--runs', type=int, default=10, help='Number of runs.')
    parser.add_argument('--epochs', type=int, default=2000, help='Max epochs.')
    parser.add_argument('--early_stopping', type=int, default=200, help='Early stopping epochs.')
    parser.add_argument('--batch_size', type=int, default=64*1024, help='Batch size.')
    parser.add_argument('--metric', type=str, default='Hits@100', help='Evaluation metric.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
    # Output and Saving
    parser.add_argument('--print_steps', type=int, default=200, help='Steps between prints.')
    parser.add_argument('--save_path', type=str, default="./results", help='Path to save results.')
    parser.add_argument('--save_res', type=bool, default=False, help='Save results flag.')
    parser.add_argument('--split', type=int, default=0, help='Save results flag.')

    args = parser.parse_args()
    args.device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')

    print("==============================")
    print(f"The arguments are: {args}")

    init_seed(args.seed)
    main(args)
