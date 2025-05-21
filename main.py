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
from model import MLP, GCN, GAT, APPNP, GPRGNN, MagNet, DGCN, DiGCN, DiGCNIB, DirGNN
from scoring import SingleClassifier, DualClassifier, InnerProduct, MLPScore, DualMLPScore
from split import link_split
from utils import get_metric_score, Logger, get_logger, get_config_dir, init_seed

log_print = get_logger('testrun', 'log', get_config_dir())


def train(model, predictor, feats, optimizer, edge_index, train_pos, train_neg, batch_size, embedding_type, criterion=None):
    model.train()
    predictor.train()
    total_loss, total_samples = 0, 0

    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        num_nodes = feats.size(0)
        device = feats.device
        emb1, emb2 = None, None

        if embedding_type == "single":
            emb1 = model(feats, edge_index)
        elif embedding_type == "dual":
            emb1, emb2 = model(feats, edge_index)

        #using BCE loss
        if criterion is None:
            pos_edge = train_pos[perm].t().to(device)
            if emb2 is None:
                pos_out = predictor(emb1[pos_edge[0]], emb1[pos_edge[1]])
            else:
                pos_out = predictor(emb1[pos_edge[0]], emb1[pos_edge[1]], emb2[pos_edge[0]], emb2[pos_edge[1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            if train_neg is None:
                #do some random sampling on every batch.
                neg_edge = negative_sampling(train_pos.t(), num_neg_samples=pos_edge.size(0), force_undirected=False)
                neg_edge = neg_edge.t().to(device)
                
            else:
                #using the fixed splitting
                neg_edge = train_neg[perm].t().to(device)
                
            if emb2 is None:
                neg_out = predictor(emb1[neg_edge[0]], emb1[neg_edge[1]])
            else:
                neg_out = predictor(emb1[neg_edge[0]], emb1[neg_edge[1]], emb2[neg_edge[0]], emb2[neg_edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            loss = pos_loss + neg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            
            num_samples = pos_out.size(0)
            total_loss += loss.item() * num_samples
            total_samples += num_samples
        
        else:
            pos_edge = train_pos[perm].t().to(device)

            if train_neg is None:
                #do some random sampling on every batch.
                neg_edge = negative_sampling(train_pos.t(), num_neg_samples=pos_edge.size(0), force_undirected=False)
                neg_edge = neg_edge.t().to(device)
            else:
                #using the fixed splitting
                neg_edge = train_neg[perm].t().to(device)

            edge = torch.cat([pos_edge, neg_edge],dim=1)
            if emb2 is None:
                out = predictor(emb1[edge[0]],emb1[edge[1]])
            else:
                out = predictor(emb1[edge[0]], emb1[edge[1]], emb2[edge[0]], emb2[edge[1]])
            
            labels = torch.cat([
                torch.ones(pos_edge.size(1), dtype=int), 
                torch.zeros(neg_edge.size(1), dtype=int)
            ]).to(device)
            
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            num_samples = pos_edge.size(0)
            total_loss += loss.item() * num_samples
            total_samples += num_samples

    return total_loss / total_samples

@torch.no_grad()
def test_edge(predictor, input_data, emb1, emb2, batch_size, device):
    preds = []
    for perm  in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t().to(device)
        if emb2 is None:
            preds += [predictor(emb1[edge[0]], emb1[edge[1]]).cpu()]
        else:
            preds += [predictor(emb1[edge[0]], emb1[edge[1]], emb2[edge[0]], emb2[edge[1]]).cpu()]
        
    pred_all = torch.cat(preds, dim=0)

    return pred_all

@torch.no_grad()
def test(model, predictor, feats, edge_index, data, batch_size, embedding_type, criterion):
    device = feats.device
    model.eval()
    predictor.eval()
    emb1, emb2 = None, None
    if embedding_type == "single":
        emb1 = model(feats, edge_index)
    elif embedding_type == "dual":
        emb1, emb2 = model(feats, edge_index)

    #using the fixed splitting to evaluate
    train_pos = data['train']['pos']
    train_neg = data['train']['neg']
    
    val_pos = data['val']['pos']
    val_neg = data['val']['neg']
    test_pos = data['test']['pos']
    test_neg = data['test']['neg']

    #pos_train_pred = test_edge(predictor, train_pos, emb1, emb2, batch_size, device)
    #neg_train_pred = test_edge(predictor, train_neg, emb1, emb2, batch_size, device)
    pos_valid_pred = test_edge(predictor, val_pos, emb1, emb2, batch_size, device)
    neg_valid_pred = test_edge(predictor, val_neg, emb1, emb2, batch_size, device)
    pos_test_pred = test_edge(predictor, test_pos, emb1, emb2, batch_size, device)
    neg_test_pred = test_edge(predictor, test_neg, emb1, emb2, batch_size, device)

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
    embedding_type = args.embedding

    run_logs = None
    if args.save_res and split < 1:
        run_logs = os.path.join(save_path,"first_run_logs.txt")

    edge_index = link_data[split]['graph'].to(device) #The observed graph for training propagation
    if args.undirected:
        edge_index = to_undirected(edge_index, data.x.shape[0])

    if args.origin_feat:
        feats = data.x.to(device)
    else:
        #use in/out degree as feature
        feats = in_out_degree(edge_index, size=data.x.shape[0]).to(device)
        print("Use in/out degree as feature!")

    args.input_dim = feats.shape[-1]

    train_pos = link_data[split]['train']['pos']
    if args.neg_sampler == "fixed":
        train_neg = link_data[split]['train']['neg']
    else:
        train_neg = None

    #building model
    model = model(args).to(device)

    # Building predictor and loss function
    if args.loss_function == "CE":
        criterion = nn.NLLLoss()
        if args.embedding == "single":
            predictor = SingleClassifier(args).to(device)
        elif args.embedding == "dual":
            predictor = DualClassifier(args).to(device)
        else:
            raise ValueError("Invalid embedding type.")
    else:
        criterion = None
        if args.embedding == "single" and args.predictor == "MLPScore":
            predictor = MLPScore(args).to(device)
        elif args.embedding == "single" and args.predictor == "InnerProduct":
            predictor = InnerProduct().to(device)
        elif args.embedding == "dual" and args.predictor == "MLPScore":
            predictor = DualMLPScore(args).to(device)
        else:
            raise ValueError("Invalid predictor type.")

    model_params = list(model.parameters())
    predictor_params = list(predictor.parameters())
    optimizer = torch.optim.Adam(model_params + predictor_params, lr=args.lr, weight_decay=args.weight_decay)
    
    best_valid = 0
    stopping_cnt = 0
    print("Run"+str(split+1)+" Start Training!")
    for epoch in range(1, args.epochs+1):
        #st_time = time.time()
        loss = train(model, predictor, feats, optimizer, edge_index, train_pos, train_neg, batch_size, embedding_type, criterion)
        results = test(model, predictor, feats, edge_index, link_data[split], batch_size, embedding_type, criterion)
        
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
    if args.net == "MLP":
        Net = MLP
    elif args.net == "GCN":
        Net = GCN
    elif args.net == "GAT":
        Net = GAT
    elif args.net == "APPNP":
        Net = APPNP
    elif args.net == "GPRGNN":
        Net = GPRGNN
    elif args.net == "MagNet":
        Net = MagNet
    elif args.net == "DGCN":
        Net = DGCN
    elif args.net == "DiGCN":
        Net = DiGCN
    elif args.net == "DiGCNIB":
        Net = DiGCNIB
    elif args.net == "DirGNN":
        Net = DirGNN
    else:
        print(f"Unsupported network type: {args.net}. Please check the available models.")

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
    parser.add_argument('--net', type=str, default='GCN', help='Model name.')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden layer size.')
    parser.add_argument('--input_dim', type=int, default=2, help='Input feature dimension.')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--nlayer', type=int, default=2, help='Number of layers.')
    parser.add_argument('--embedding', type=str, default="single", help='Embedding type.')
    # Hyperparameters
    parser.add_argument('--dropout', type=float, default=0.5, help='DropouACt rate.')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads for GAT.')
    parser.add_argument('--output_heads', type=int, default=1, help='Number of attention output head for GAT.')
    parser.add_argument('--K', type=int, default=1, help='Parameter K.')
    parser.add_argument('--q', type=float, default=0.20, help='Hyperparameter q.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Hyperparameter alpha.')
    parser.add_argument('--jk', type=str, default="cat", help='Jumping Knowledge method.')
    parser.add_argument('--normalize', type=bool, default=False, help='Use nomalization for DirGNN.')
    # Predictor
    parser.add_argument('--predictor', type=str, default="MLPScore", help='Predictor type.')
    parser.add_argument('--nlayer_predictor', type=int, default=2, help='Predictor layers.')
    parser.add_argument('--hidden_predictor', type=int, default=64, help='Predictor hidden layer size.')
    parser.add_argument('--outdim_predictor', type=int, default=2, help='Predictor output dim.')
    parser.add_argument('--operator', type=str, default="hadamard", help='Predictor opertor type, hadamard/cat.')
    # Training
    parser.add_argument('--runs', type=int, default=10, help='Number of runs.')
    parser.add_argument('--epochs', type=int, default=2000, help='Max epochs.')
    parser.add_argument('--early_stopping', type=int, default=200, help='Early stopping epochs.')
    parser.add_argument('--batch_size', type=int, default=64*1024, help='Batch size.')
    parser.add_argument('--metric', type=str, default='Hits@100', help='Evaluation metric.')
    parser.add_argument('--loss_function', type=str, default="BCE", help='Loss function.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
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
