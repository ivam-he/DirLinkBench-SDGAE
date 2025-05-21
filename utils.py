import torch
import os
import sys
import logging
import warnings
import random
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from evaluation import evaluate_acc, evaluate_auc, evaluate_hits, evaluate_mrr
from torch_geometric.utils import to_scipy_sparse_matrix
import psutil

def init_seed(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config_dir():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(file_dir, "config")

def get_metric_score(pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    result = {}

    ##Hits@K
    k_list = [20, 50, 100]
    #2result_hit_train = evaluate_hits(pos_train_pred, neg_train_pred, k_list)
    result_hit_val = evaluate_hits(pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(pos_test_pred, neg_test_pred, k_list)
    for K in [20, 50, 100]:
        #2result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])
        result[f'Hits@{K}'] = (0.0, result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])

    ##MRR
    #result_mrr_train = evaluate_mrr(pos_train_pred, neg_train_pred.repeat(pos_train_pred.size(0), 1))
    #result_mrr_val = evaluate_mrr(pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1) )
    #result_mrr_test = evaluate_mrr(pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1) )

    #2result_mrr_train = evaluate_mrr(pos_train_pred, neg_train_pred)
    result_mrr_val = evaluate_mrr(pos_val_pred, neg_val_pred)
    result_mrr_test = evaluate_mrr(pos_test_pred, neg_test_pred)
    #2result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    result['MRR'] = (0.0, result_mrr_val['MRR'], result_mrr_test['MRR'])

    ##AUC&AP
    #2train_pred = torch.cat([pos_train_pred, neg_train_pred])
    #2train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
    #2                        torch.zeros(neg_train_pred.size(0), dtype=int)])
    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])

    #2result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    #2result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AUC'] = (0.0, result_auc_val['AUC'], result_auc_test['AUC'])
    #2result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])
    result['AP'] = (0.0, result_auc_val['AP'], result_auc_test['AP'])

    ##ACC
    #2train_pred = (train_pred>=0.5).int()
    val_pred = (val_pred>=0.5).int()
    test_pred = (test_pred>=0.5).int()
    #2result_acc_train = evaluate_acc(train_pred, train_true)
    result_acc_val = evaluate_acc(val_pred, val_true)
    result_acc_test = evaluate_acc(test_pred, test_true)
    #2result['ACC'] = (result_acc_train, result_acc_val, result_acc_test)
    result['ACC'] = (0.0, result_acc_val, result_acc_test)
    
    return result

def GraphInfo(data):
    edge_index = data.edge_index
    x = data.x

    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]
    dim_feat  = x.shape[1]

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Dimension of feature: {dim_feat}")

    # Determine if the graph is directed
    row, col = edge_index
    row = row.numpy()
    col = col.numpy()
    row_min = np.minimum(row, col)
    col_max = np.maximum(row, col)
    edges_canonical = np.stack((row_min, col_max), axis=1)
    edges_canonical_view = edges_canonical.view([('u', row_min.dtype), ('v', col_max.dtype)])
    nique_edges, counts = np.unique(edges_canonical_view, return_counts=True)
    undirected_edges = np.sum(counts == 2) * 2
    directed_edges = np.sum(counts == 1)
    total_edges = directed_edges + undirected_edges
    assert num_edges == total_edges
    
    if directed_edges == 0:
        print("The graph is undirected.")
    else:
        print("The graph is directed.")
        print(f"Number of directed edges: {directed_edges}")
        print(f"Number of undirected edges: {undirected_edges}")
        proportion_directed = directed_edges / total_edges
        print(f"Proportion of directed edges: {proportion_directed:.2%}")

    # Check if the graph is connected
    adj_mat=sp.csr_matrix((np.ones(row.shape[0]),(row,col)),shape=(num_nodes,num_nodes))

    if directed_edges == 0:
        n_components, labels = connected_components(csgraph=adj_mat, directed=True, connection='weak',return_labels=True)
    else:
        n_components, labels = connected_components(csgraph=adj_mat, directed=False,return_labels=True)

    if n_components == 1:
        print("The graph is connected.")
    else:
        print("The graph is not connected.")
        print(f"The Number of components is: {n_components}")

    # Determine whether there are duplicate edges
    N = torch.max(edge_index) + 1  
    u = edge_index[0].to(torch.int64)
    v = edge_index[1].to(torch.int64)
    edge_hashes = u * N + v

    sorted_hashes, indices = torch.sort(edge_hashes)
    unique_hashes = torch.unique_consecutive(sorted_hashes)

    u_unique = unique_hashes // N
    v_unique = unique_hashes % N

    edge_index_unique = torch.stack([u_unique, v_unique], dim=0)
    data.edge_index = edge_index_unique
    num_duplicates = num_edges - edge_index_unique.size(1)
    if num_duplicates > 0:
        print(f"There are duplicate edges, the number is {num_duplicates}")
    else:
        print("No duplicate edges.")

    # Check if 'x' has been processed by T.NormalizeFeatures()
    #if torch.all(x >= 0):
    #    row_sums = x.sum(dim=-1)
    #    if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
    #        print("The feature matrix 'x' has been processed by T.NormalizeFeatures().")
    #    else:
    #        print("The feature matrix 'x' has not been processed by T.NormalizeFeatures().")
    #else:
    #    print("The feature matrix 'x' has not been processed by T.NormalizeFeatures().")

    print("==============================")


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    #def save_result(self, save_path):
    #    result_file = os.path.join(save_path,"result.excel")


    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')

        else:
            best_results = []

            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')

            #r = best_result[:, 0].float()
            #print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 1].float()
            best_valid_mean = round(r.mean().item(), 2)
            best_valid_var = round(r.std().item(), 2)

            best_valid = str(best_valid_mean) +' ' + '±' +  ' ' + str(best_valid_var)
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')


            #r = best_result[:, 2].float()
            #best_train_mean = round(r.mean().item(), 2)
            #best_train_var = round(r.std().item(), 2)
            #final_train = str(best_train_mean) +' ' + '±' +  ' ' + str(best_train_var)
            #print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 3].float()
            best_test_mean = round(r.mean().item(), 2)
            best_test_var = round(r.std().item(), 2)
            final_test = str(best_test_mean) +' ' + '±' +  ' ' + str(best_test_var)
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            #mean_list = [best_train_mean, best_valid_mean, best_test_mean]
            #var_list = [best_train_var, best_valid_var, best_test_var]


            return best_result, best_valid,  final_test


def get_logger(name, log_dir, config_dir):
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

def sqrtinvdiag(M: sp.spmatrix) -> sp.csc_matrix:
    """Inverts and square-roots a positive diagonal matrix.
    Args:
        M (scipy sparse matrix): matrix to invert
    Returns:
        scipy sparse matrix of inverted square-root of diagonal
    """
    d = M.diagonal()
    dd = [1 / max(np.sqrt(x), 1 / 999999999) for x in d]
    return sp.dia_matrix((dd, [0]), shape=(len(d), len(d))).tocsc()

def separate_positive_negative(num_nodes, edge_index, edge_weight):
    ind = edge_weight > 0
    edge_index_p = edge_index[:, ind]
    edge_weight_p = edge_weight[ind]
    ind = edge_weight < 0
    edge_index_n = edge_index[:, ind]
    edge_weight_n = - edge_weight[ind]
    A_p = to_scipy_sparse_matrix(edge_index_p, edge_weight_p, num_nodes=num_nodes)
    A_n = to_scipy_sparse_matrix(edge_index_n, edge_weight_n, num_nodes=num_nodes)
    return edge_index_p, edge_weight_p, edge_index_n, edge_weight_n, A_p, A_n


def set_spectral_adjacency_reg_features(num_nodes, edge_index, edge_weight, k=2, normalization=None, tau_p=None, tau_n=None, eigens=None, mi=None):
    """generate the graph features using eigenvectors of the regularised adjacency matrix.
    """
    print(f"Generate {k}-dimension node feature matrix based on the structure topology, it may take a while...")
    edge_index_p, edge_weight_p, edge_index_n, edge_weight_n, A_p, A_n = separate_positive_negative(num_nodes, edge_index, edge_weight)
    A = (A_p - A_n).tocsc()
    A_p = sp.csc_matrix(A_p)
    A_n = sp.csc_matrix(A_n)
    D_p = sp.diags(A_p.sum(axis=0).tolist(), [0]).tocsc()
    D_n = sp.diags(A_n.sum(axis=0).tolist(), [0]).tocsc()
    Dbar = (D_p + D_n)
    d = sqrtinvdiag(Dbar)
    size = A_p.shape[0]
    if eigens == None:
        eigens = k
    if mi == None:
        mi = size
    if tau_p == None or tau_n == None:
        tau_p = 0.25 * np.mean(Dbar.data) / size
        tau_n = 0.25 * np.mean(Dbar.data) / size
    p_tau = A_p.copy().astype(np.float32)
    n_tau = A_n.copy().astype(np.float32)
    p_tau.data += tau_p
    n_tau.data += tau_n
    Dbar_c = size - Dbar.diagonal()
    Dbar_tau_s = (p_tau + n_tau).sum(axis=0) + (Dbar_c * abs(tau_p - tau_n))[None, :]  
    Dbar_tau = sp.diags(Dbar_tau_s.tolist(), [0])
    if normalization is None:
        matrix = A
        delta_tau = tau_p - tau_n
        def mv(v):
            return matrix.dot(v) + delta_tau * v.sum()
    elif normalization == 'sym':
        d = sqrtinvdiag(Dbar_tau)
        matrix = d * A * d
        dd = d.diagonal()
        tau_dd = (tau_p - tau_n) * dd
        def mv(v):
            return matrix.dot(v) + tau_dd * dd.dot(v)
    elif normalization == 'sym_sep':
        diag_corr = sp.diags([size * tau_p] * size).tocsc()
        dp = sqrtinvdiag(D_p + diag_corr)
        matrix = dp * A_p * dp
        diag_corr = sp.diags([size * tau_n] * size).tocsc()
        dn = sqrtinvdiag(D_n + diag_corr)
        matrix = matrix - (dn * A_n * dn)
        dpd = dp.diagonal()
        dnd = dn.diagonal()
        tau_dp = tau_p * dpd
        tau_dn = tau_n * dnd
        def mv(v):
            return matrix.dot(v) + tau_dp * dpd.dot(v) - tau_dn * dnd.dot(v)
    else:
        raise NameError('Error in choosing normalization!')

    matrix_o = sp.linalg.LinearOperator(matrix.shape, matvec=mv)
    (w, v) = sp.linalg.eigs(matrix_o, int(eigens), maxiter=mi, which='LR')
    v = v * w  # weight eigenvalues by eigenvectors, since larger eigenvectors are more likely to be informative
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = torch.FloatTensor(v)

    #nomorlization
    features = F.normalize(features, p=1, dim=1)
    features = features.numpy()
    m = features.mean(axis=0)
    s = features.std(axis=0, ddof=0, keepdims=True) + 1e-12
    features -= m
    features /= s
    return torch.FloatTensor(features)







