import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import average_precision_score
from ogb.linkproppred import Evaluator

"""
This code benefits from Open Graph Benchmark(https://ogb.stanford.edu/)
"""

def evaluate_acc(labels, pred_labels):
    labels = labels.detach().cpu()
    pred_labels = pred_labels.detach().cpu()
    acc = accuracy_score(labels, pred_labels)
    acc = round(acc, 4)

    return acc


def evaluate_auc(val_pred, val_true):
    results = {}
    valid_auc = roc_auc_score(val_true, val_pred)
    valid_auc = round(valid_auc, 4)
    results['AUC'] = valid_auc

    valid_ap = average_precision_score(val_true, val_pred)
    valid_ap = round(valid_ap, 4)
    results['AP'] = valid_ap

    return results
        

def evaluate_hits(pos_pred, neg_pred, k_list):
    evaluator = Evaluator(name="ogbl-collab")
    results = {}  
    for K in k_list:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']

        hits = round(hits, 4)
        results[f'Hits@{K}'] = hits

    return results


'''def evaluate_mrr(pos_val_pred, neg_val_pred):
    
    #    compute mrr
    #    neg_val_pred is an array with shape (batch size, num_entities_neg).
    #    pos_val_pred is an array with shape (batch size, )
    
    results = {} 
    print(pos_val_pred.shape)
    print(neg_val_pred.shape)

    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)

    # calculate ranks
    pos_val_pred= pos_val_pred.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (neg_val_pred >= pos_val_pred).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (neg_val_pred > pos_val_pred).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    mrr_list = 1./ranking_list.to(torch.float)
    valid_mrr =mrr_list.mean().item()
    valid_mrr = round(valid_mrr, 4)
    results['MRR'] = valid_mrr 

    return results'''


def evaluate_mrr(pos_pred, neg_pred):
    """
    Mean Reciprocal Rank (MRR)
    
    Args:
        pos_pred (Tensor): Prediction scores for positive samples
        neg_pred (Tensor): Prediction scores for negative samples
    
    Returns:
        dict: MRR dict
    """
    results = {}
    neg_pred = neg_pred.view(-1)
    pos_pred = pos_pred.view(-1)

    #print(pos_pred.shape)
    #print(neg_pred.shape)
    
    # ascending ordering of negative sample predictions
    sorted_neg, _ = torch.sort(neg_pred)
    M = sorted_neg.size(0)
    
    # Use searchsorted to find the insertion position of each pos_pred
    # ‘right=False’ means find the first index >= pos_pred
    # ‘right=True’ means find the first index > pos_pred
    idx_ge = torch.searchsorted(sorted_neg, pos_pred, right=False)
    idx_gt = torch.searchsorted(sorted_neg, pos_pred, right=True)
    
    num_ge = M - idx_ge
    num_gt = M - idx_gt
    
    ranking = 0.5 * (num_ge + num_gt) + 1.0
    
    # MRR
    mrr = torch.mean(1.0 / ranking).item()
    mrr = round(mrr, 4)
    results['MRR'] = mrr
    
    return results

