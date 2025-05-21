import torch
import sys
import os
import argparse
import numpy as np
import pandas as pd
from data_loader import dataloader
from split import link_split
from evaluation import evaluate_acc, evaluate_auc, evaluate_hits, evaluate_mrr
from scipy.special import expit 
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser(description="parameters")
# Dataset
parser.add_argument('--num_embed', type=int, default=60, help='Number of nodes.')
parser.add_argument('--dataset', type=str, default='cora_ml', help='Dataset name.')
parser.add_argument('--prob_val', type=float, default=0.05, help='Validation set proportion.')
parser.add_argument('--prob_test', type=float, default=0.15, help='Test set proportion.')
parser.add_argument('--runs', type=int, default=10, help='Number of runs.')
parser.add_argument('--predictor', type=str, default="Logistic")
parser.add_argument('--operator', type=str, default="cat")
parser.add_argument('--emb_file', type=str, default='./odin/embedding/', help='Embeddings output filename.')
args = parser.parse_args()

def get_metric_score(pos_test_pred, neg_test_pred):
    result = {}
    ##Hits@K
    k_list = [20, 50, 100]
    result_hit_test = evaluate_hits(pos_test_pred, neg_test_pred, k_list)
    for K in [20, 50, 100]:
        result[f'Hits@{K}'] = (result_hit_test[f'Hits@{K}'])
    ##MRR
    result_mrr_test = evaluate_mrr(pos_test_pred, neg_test_pred)
    result['MRR'] = (result_mrr_test['MRR'])
    ##AUC&AP
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), torch.zeros(neg_test_pred.size(0), dtype=int)])
    result_auc_test = evaluate_auc(test_pred, test_true)
    result['AUC'] = (result_auc_test['AUC'])
    result['AP'] = (result_auc_test['AP'])
    ##ACC
    test_pred = (test_pred>=0.5).int()
    result_acc_test = evaluate_acc(test_pred, test_true)
    result['ACC'] = (result_acc_test)
    return result

def logistic_regression_classifier(train_pos, train_neg, test_pos, test_neg, embedding_s, embedding_t, operator):
    
    def compute_embeddings(edges, operator):
        if edges.size == 0:
            return np.array([])
        sources = edges[:, 0]
        targets = edges[:, 1]

        emb_s = embedding_s[sources]
        emb_t = embedding_t[targets]

        if operator == "cat":
            return np.concatenate((emb_s, emb_t), axis=1)
        elif operator == "hadamard":
            return emb_s * emb_t
        elif operator == "cat2": #used in odin-released codes
            emb_s2 = embedding_t[sources]
            emb_t2 = embedding_s[targets]
            return np.concatenate((emb_s, emb_s2, emb_t, emb_t2), axis=1)
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    train_edges = np.vstack((train_pos, train_neg))
    X_train = compute_embeddings(train_edges, operator)
    Y_train = np.concatenate((np.ones(len(train_pos)), np.zeros(len(train_neg))))

    test_edges = np.vstack((test_pos, test_neg))
    X_test = compute_embeddings(test_edges, operator)

    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_train, Y_train)

    test_y_score = lr.predict_proba(X_test)[:, 1]
    test_pos_scores = test_y_score[:len(test_pos)]
    test_neg_scores = test_y_score[len(test_pos):]

    return test_pos_scores, test_neg_scores

def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def main(args):
    dataset = dataloader(args.dataset)
    data = dataset[0]
    num_node = data.x.shape[0]

    #data splitting
    link_data = link_split(
        name=args.dataset, 
        data=data, 
        num_splits=args.runs, 
        prob_test=args.prob_test, 
        prob_val=args.prob_val,  
        maintain_connect=True
    )
    results = {}
    results['Hits@20'] = []
    results['Hits@50'] = []
    results['Hits@100'] = []
    results['MRR'] = []
    results['AUC'] = []
    results['AP'] = []
    results['ACC'] = []
    emb_file_0 = args.emb_file
    for split in range(args.runs):
        
        emb_file = os.path.join(emb_file_0, args.dataset)
        emb_file = os.path.join(emb_file, str(split))

        #using the fixed splitting to evaluate
        train_pos = link_data[split]['train']['pos'].tolist()
        train_neg = link_data[split]['train']['neg'].tolist()
        test_pos = link_data[split]['test']['pos'].tolist()
        test_neg = link_data[split]['test']['neg'].tolist()

        num_embed = args.num_embed
        embedding_s = np.zeros((num_node, num_embed)) #np.random.rand(n_node, n_embed)
        embedding_t = np.zeros((num_node, num_embed)) #np.random.rand(n_node, n_embed)
        
        #Read embeddings
        emb_file_s = os.path.join(emb_file, "source.emb")
        emb_file_t = os.path.join(emb_file, "target.emb")
        with open(emb_file_s, "r") as f:
            lines = f.readlines()
            lines = lines[1:]  # skip the first line
            for line in lines:
                emd = line.split()
                embedding_s[int(emd[0]), :] = str_list_to_float(emd[1:])

        with open(emb_file_t, "r") as f:
            lines = f.readlines()
            lines = lines[1:]  # skip the first line
            for line in lines:
                emd = line.split()
                embedding_t[int(emd[0]), :] = str_list_to_float(emd[1:])

        if args.predictor == "Logistic":
            pos_test_pred,  neg_test_pred = logistic_regression_classifier(train_pos, train_neg, test_pos, test_neg, embedding_s, embedding_t, args.operator)

        elif args.predictor == "InnerProduct":
            test_pos = np.array(test_pos)
            test_neg = np.array(test_neg)

            sources_pos = test_pos[:, 0]
            targets_pos = test_pos[:, 1]
            scores_pos  = np.sum(embedding_s[sources_pos] * embedding_t[targets_pos], axis=1)
            pos_test_pred = expit(scores_pos)

            sources_neg = test_neg[:, 0]
            targets_neg = test_neg[:, 1]
            scores_neg  = np.sum(embedding_s[sources_neg] * embedding_t[targets_neg], axis=1)          
            neg_test_pred = expit(scores_neg)

        pos_test_pred = torch.from_numpy(pos_test_pred)
        neg_test_pred = torch.from_numpy(neg_test_pred)
        result = get_metric_score(pos_test_pred, neg_test_pred)

        for key, res in result.items():
            results[key].append(res)
    #print(results)

    df = pd.DataFrame(results)
    mean_values = df.mean()*100
    std_values = df.std()*100
    summary = pd.DataFrame({
        'mean': mean_values,
        'std': std_values
    })
    summary_formatted = summary.apply(lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}", axis=1)
    summary_formatted = summary_formatted.reset_index()
    summary_formatted.columns = ['metric', 'mean ± std']
    print(summary_formatted.to_string(index=False))


if __name__ == "__main__":
    main(args)
