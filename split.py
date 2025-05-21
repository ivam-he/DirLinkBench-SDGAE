import torch
import scipy
import numpy as np
import os
import os.path as osp
import networkx as nx
import scipy.sparse as sp
from networkx.algorithms import tree
import torch_geometric
from torch_geometric.utils import negative_sampling, to_undirected, to_scipy_sparse_matrix
from scipy.sparse import coo_matrix

def link_split(name, data, num_splits, prob_test, prob_val, seed=1, maintain_connect=True, root = "./split"):
    name = name.lower()
    edge_index = data.edge_index.cpu() #directed graph A
    num_nodes = data.x.shape[0]
    num_edges = edge_index.shape[1]

    num_val = int(prob_val*num_edges)
    num_test = int(prob_test*num_edges)
    num_train = num_edges-num_val-num_test

    edge_index_u = to_undirected(edge_index) 
    A_u = to_scipy_sparse_matrix(edge_index_u) #undirected A_u = A Union A^T

    if maintain_connect:
        G = nx.from_scipy_sparse_matrix(A_u, create_using=nx.Graph, edge_attribute='weight')
        mst_tmp = list(tree.minimum_spanning_edges(G, algorithm="kruskal", data=False)) #the minimum spanning tree of A_u
        
        all_edges_set = set(map(tuple, edge_index.T.tolist()))
        mst_r = [t[::-1] for t in mst_tmp] #the reverse edge
        mst_tmp += mst_r
        mst = [edge for edge in mst_tmp if edge in all_edges_set] #filter the edges of the mst from all edges
        nmst = list(all_edges_set - set(mst)) #mst denote the edges in minimum spanning tree, nmst is that not in.

        if len(nmst) < (num_val+num_test):
            raise ValueError("There are no enough edges to be removed for validation/testing. Please use a smaller prob_test or prob_val.")
    else:
        mst = []
        nmst = edge_index.T.tolist()
    assert len(mst)+len(nmst) == num_edges #check

    datasets = {}
    tmp_print = True
    tmp_check = True
    for ind in range(num_splits):
        seed = seed+ind
        path = osp.join(root, name, str(ind))
        if os.path.exists(path+"/observed_graph.npy"):
            if tmp_print:
                print("Load existing data splits.")
            tmp_print = False
            pos_train_edges = np.load(path+"/pos_train_edges.npy")
            neg_train_edges = np.load(path+"/neg_train_edges.npy")

            pos_val_edges = np.load(path+"/pos_val_edges.npy")
            neg_val_edges = np.load(path+"/neg_val_edges.npy")

            pos_test_edges = np.load(path+"/pos_test_edges.npy")
            neg_test_edges = np.load(path+"/neg_test_edges.npy")

            observed_graph = np.load(path+"/observed_graph.npy")
        else:
            #ramdom split the positive edges
            rs = np.random.RandomState(seed)
            rs.shuffle(nmst)
            pos_test_edges = nmst[:num_test]
            pos_val_edges = nmst[num_test:num_test+num_val]
            pos_train_edges = nmst[num_test+num_val:]+mst

            pos_test_edges = np.array(list(map(list,pos_test_edges)))
            pos_val_edges = np.array(list(map(list,pos_val_edges)))
            pos_train_edges = np.array(list(map(list,pos_train_edges)))

            #get obeserved graph and check it is in A
            observed_graph = pos_train_edges.T
            if tmp_check:
                observed_graph_check = torch.from_numpy(observed_graph).long()
                max_node = max(edge_index.max().item(), observed_graph_check.max().item())
                edge_ids = edge_index[0] * (max_node + 1) + edge_index[1]  # shape: [num_edges]
                observed_ids = observed_graph_check[0] * (max_node + 1) + observed_graph_check[1]  # shape: [num_observed_edges]
                is_in_adj = torch.isin(observed_ids, edge_ids).all().item()
                assert is_in_adj
                tmp_check = False

            #negative sampling
            '''Random sample training negative samples in the observed graph condition'''
            neg_train_edges = negative_sampling(torch.from_numpy(observed_graph).long(), num_neg_samples=num_train, force_undirected=False)
            neg_train_edges = neg_train_edges.numpy().T

            '''Random sample validation and testing negative samples in the all graph condition'''
            neg_test_val_edges = negative_sampling(edge_index, num_neg_samples=num_test+num_val, force_undirected=False)
            neg_test_val_edges = neg_test_val_edges.numpy().T
            neg_test_val_edges = map(tuple, neg_test_val_edges)
            neg_test_val_edges = list(neg_test_val_edges)
            neg_test_edges = neg_test_val_edges[:num_test]
            neg_val_edges = neg_test_val_edges[num_test:]

            neg_test_edges = np.array(list(map(list,neg_test_edges)))
            neg_val_edges = np.array(list(map(list,neg_val_edges)))

            #save
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(path+"/pos_test_edges.npy",pos_test_edges)
            np.save(path+"/neg_test_edges.npy",neg_test_edges)
            np.save(path+"/pos_val_edges.npy",pos_val_edges)
            np.save(path+"/neg_val_edges.npy",neg_val_edges)
            np.save(path+"/pos_train_edges.npy",pos_train_edges)
            np.save(path+"/neg_train_edges.npy",neg_train_edges)
            np.save(path+"/observed_graph.npy",observed_graph)

        datasets[ind] = {}
        datasets[ind]['graph'] = torch.from_numpy(observed_graph).long()
        datasets[ind]['train'] = {}
        datasets[ind]['train']['pos'] = torch.from_numpy(pos_train_edges).long()
        datasets[ind]['train']['neg'] = torch.from_numpy(neg_train_edges).long()
        datasets[ind]['val'] = {}
        datasets[ind]['val']['pos'] = torch.from_numpy(pos_val_edges).long()
        datasets[ind]['val']['neg'] = torch.from_numpy(neg_val_edges).long()
        datasets[ind]['test'] = {}
        datasets[ind]['test']['pos'] = torch.from_numpy(pos_test_edges).long()
        datasets[ind]['test']['neg'] = torch.from_numpy(neg_test_edges).long()
    
    return datasets



    








