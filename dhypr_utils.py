import os
import torch
import scipy.sparse as sp
import pdb
import os.path as osp
import numpy as np
import networkx as nx
import time
import pickle

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

K_max = 2

def compute_proximity_matrices(adj, K=2):
    A = dict()
    for k in range(1, K+1):
        t = time.time()
        compute_kth_diffusion_in(adj, k, A) 
        print('k={} {}   took {} s'.format(k, 'diffusion_in', time.time()-t))
        
        t = time.time()
        compute_kth_diffusion_out(adj, k, A)
        print('k={} {}   took {} s'.format(k, 'diffusion_out', time.time()-t))
        
        t = time.time()
        compute_kth_neighbor_in(adj, k, A)
        print('k={} {}   took {} s'.format(k, 'neighbor_in', time.time()-t))
        
        t = time.time()
        compute_kth_neighbor_out(adj, k, A)
        print('k={} {}   took {} s'.format(k, 'neighbor_out', time.time()-t))
    
    return A

def compute_kth_diffusion_in(adj, k, A):
    if k == 1:
        A['a'+str(k)+'_d_i'] = adj.T
        
    if k > 1:
        A['a'+str(k)+'_d_i'] = np.where(np.dot(A['a'+str(k-1)+'_d_i'], A['a1_d_i']) > 0, 1, 0) 
    return 

def compute_kth_diffusion_out(adj, k, A):
    if k == 1:
        A['a'+str(k)+'_d_o'] = adj
        
    if k > 1:
        A['a'+str(k)+'_d_o'] = np.where(np.dot(A['a'+str(k-1)+'_d_o'], A['a1_d_o']) > 0, 1, 0) 
    return 

def compute_kth_neighbor_in(adj, k, A):
    tmp = np.dot(A['a'+str(k)+'_d_i'], A['a'+str(k)+'_d_o'])
    np.fill_diagonal(tmp, 0) 
    A['a'+str(k)+'_n_i'] = np.where(tmp + tmp.T - np.diag(tmp.diagonal()) > 0, 1, 0) 
    return 
    
def compute_kth_neighbor_out(adj, k, A):
    tmp = np.dot(A['a'+str(k)+'_d_o'], A['a'+str(k)+'_d_i'])
    np.fill_diagonal(tmp, 0) 
    A['a'+str(k)+'_n_o'] = np.where(tmp + tmp.T - np.diag(tmp.diagonal()) > 0, 1, 0) 
    return 


def get_our_k_order_matrix(edge_index):
    #G = nx.read_edgelist(datapath, delimiter='\t', create_using=nx.DiGraph())
    edge_index = edge_index.numpy()
    edges = zip(edge_index[0], edge_index[1])
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    adj = nx.adjacency_matrix(G).toarray()

    A = compute_proximity_matrices(adj, K=K_max)
    A = {key: sp.csr_matrix(A[key]) for key in A}

    #with open(os.path.join(datadir, 'train_graph_kth_order_matrices.pickle'), 'wb') as f:
    #    pickle.dump(A, f)
    return A

def get_processed_adj(edge_index, normalize_adj):
    data = get_our_k_order_matrix(edge_index)
    data['adj_train_norm'] = dict()
    
    if normalize_adj:
        if 'adj_train' in data:
            data['adj_train_norm']['adj_train_norm'] = sparse_mx_to_torch_sparse_tensor(
                    normalize(data['adj_train'] + sp.eye(data['adj_train'].shape[0])))
        if 'a1_d_i' in data:
            data['adj_train_norm']['a1_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a1_d_i'] + sp.eye(data['a1_d_i'].shape[0])))
        if 'a1_d_o' in data:
            data['adj_train_norm']['a1_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a1_d_o'] + sp.eye(data['a1_d_o'].shape[0])))
        if 'a1_n_i' in data:
            data['adj_train_norm']['a1_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a1_n_i'] + sp.eye(data['a1_n_i'].shape[0])))
        if 'a1_n_o' in data:
            data['adj_train_norm']['a1_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a1_n_o'] + sp.eye(data['a1_n_o'].shape[0])))
        if 'a2_d_i' in data:
            data['adj_train_norm']['a2_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a2_d_i'] + sp.eye(data['a2_d_i'].shape[0])))
        if 'a2_d_o' in data:
            data['adj_train_norm']['a2_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a2_d_o'] + sp.eye(data['a2_d_o'].shape[0])))
        if 'a2_n_i' in data:
            data['adj_train_norm']['a2_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a2_n_i'] + sp.eye(data['a2_n_i'].shape[0])))
        if 'a2_n_o' in data:
            data['adj_train_norm']['a2_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a2_n_o'] + sp.eye(data['a2_n_o'].shape[0])))
        if 'a3_d_i' in data:
            data['adj_train_norm']['a3_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a3_d_i'] + sp.eye(data['a3_d_i'].shape[0])))
        if 'a3_d_o' in data:
            data['adj_train_norm']['a3_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a3_d_o'] + sp.eye(data['a3_d_o'].shape[0])))
        if 'a3_n_i' in data:
            data['adj_train_norm']['a3_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a3_n_i'] + sp.eye(data['a3_n_i'].shape[0])))
        if 'a3_n_o' in data:
            data['adj_train_norm']['a3_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a3_n_o'] + sp.eye(data['a3_n_o'].shape[0])))
    else:
        if 'adj_train' in data:
            data['adj_train_norm']['adj_train_norm'] = sparse_mx_to_torch_sparse_tensor(
                    data['adj_train'])
        if 'a1_d_i' in data:
            data['adj_train_norm']['a1_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                    data['a1_d_i'])
        if 'a1_d_o' in data:
            data['adj_train_norm']['a1_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                    data['a1_d_o'])
        if 'a1_n_i' in data:
            data['adj_train_norm']['a1_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                    data['a1_n_i'])
        if 'a1_n_o' in data:
            data['adj_train_norm']['a1_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a1_n_o'])
        if 'a2_d_i' in data:
            data['adj_train_norm']['a2_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a2_d_i'])
        if 'a2_d_o' in data:
            data['adj_train_norm']['a2_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a2_d_o'])
        if 'a2_n_i' in data:
            data['adj_train_norm']['a2_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a2_n_i'])
        if 'a2_n_o' in data:
            data['adj_train_norm']['a2_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a2_n_o'])
        if 'a3_d_i' in data:
            data['adj_train_norm']['a3_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a3_d_i'])
        if 'a3_d_o' in data:
            data['adj_train_norm']['a3_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a3_d_o'])
        if 'a3_n_i' in data:
            data['adj_train_norm']['a3_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a3_n_i'])
        if 'a3_n_o' in data:
            data['adj_train_norm']['a3_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a3_n_o'])
            
    return data['adj_train_norm']

 