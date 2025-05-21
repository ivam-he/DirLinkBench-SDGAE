import torch
import math
import pickle
import os.path as osp
import os
import numpy as np
import json
from itertools import chain
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix, remove_self_loops
import scipy.sparse as sp
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from typing import Callable, Optional
from torch_geometric.io import read_npz
#from utils import set_spectral_adjacency_reg_features

class Citation(InMemoryDataset):
    url = ('https://github.com/gasteigerjo/ppnp/tree/master/ppnp/data')
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cora_ml','citeseer']
        super(Citation, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [self.name+'.npz']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self):
        data = np.load(self.raw_paths[0], allow_pickle=True)
        data_dict=(dict(data))
        
        init_dict = {}
        del_entries = []

        # Construct sparse matrices
        for key in data_dict.keys():
            if key.endswith('_data') or key.endswith('.data'):
                sep = '.'
                matrix_name = key[:-5]
                mat_data = key
                mat_indices = '{}{}indices'.format(matrix_name, sep)
                mat_indptr = '{}{}indptr'.format(matrix_name, sep)
                mat_shape = '{}{}shape'.format(matrix_name, sep)
                init_dict[matrix_name] = sp.csr_matrix((data_dict[mat_data],data_dict[mat_indices],data_dict[mat_indptr]),shape=data_dict[mat_shape])
                del_entries.extend([mat_data, mat_indices, mat_indptr, mat_shape])

        # Delete sparse matrix entries
        for del_entry in del_entries:
            del data_dict[del_entry]

        # Load everything else
        for key, val in data_dict.items():
            if ((val is not None) and (None not in val)):
                init_dict[key] = val

        x = torch.FloatTensor(init_dict["attr_matrix"].toarray())
        y = torch.LongTensor(init_dict["labels"])
        adj = init_dict["adj_matrix"].tocoo()
        indices = np.array([adj.row, adj.col])
        edge_index = torch.LongTensor(indices)

        #Link prediction tasks do not require category labels
        data = Data(x=x, edge_index=edge_index)

        ###Select the largest connected components in the graph
        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        num_components, component = sp.csgraph.connected_components(adj)
        _, count = np.unique(component, return_counts=True)
        subset = np.in1d(component, count.argsort()[-1:])
        data=data.subgraph(torch.from_numpy(subset).to(torch.bool))

        ###remove self-loops
        data.edge_index,_=remove_self_loops(data.edge_index)

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class Amazon(InMemoryDataset):
    url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        assert self.name in ['computers', 'photo']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'amazon_electronics_{self.name.lower()}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url + self.raw_file_names, self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0], to_undirected=False)
        #Link prediction tasks do not require category labels
        del data.y

        ###Select the largest connected components in the graph
        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        num_components, component = sp.csgraph.connected_components(adj)
        _, count = np.unique(component, return_counts=True)
        subset = np.in1d(component, count.argsort()[-1:])
        data=data.subgraph(torch.from_numpy(subset).to(torch.bool))

        ###remove self-loops
        data.edge_index,_=remove_self_loops(data.edge_index)

        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name.capitalize()}()'


class WikiCS(InMemoryDataset):
    r"""This is the copy of the torch_geometric.datasets.WikiCS (v1.6.3)

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.url = 'https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset'
        super(WikiCS, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.json']

    @property
    def processed_file_names(self):
        return 'wikics.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = json.load(f)

        x = torch.tensor(data['features'], dtype=torch.float)
        y = torch.tensor(data['labels'], dtype=torch.long)

        edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
        edges = list(chain(*edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        train_mask = torch.tensor(data['train_masks'], dtype=torch.bool)
        train_mask = train_mask.t().contiguous()

        val_mask = torch.tensor(data['val_masks'], dtype=torch.bool)
        val_mask = val_mask.t().contiguous()

        test_mask = torch.tensor(data['test_mask'], dtype=torch.bool)

        stopping_mask = torch.tensor(data['stopping_masks'], dtype=torch.bool)
        stopping_mask = stopping_mask.t().contiguous()

        #data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
        #            val_mask=val_mask, test_mask=test_mask,
        #            stopping_mask=stopping_mask)
        data = Data(x=x, edge_index=edge_index)

        ###Select the largest connected components in the graph
        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        num_components, component = sp.csgraph.connected_components(adj)
        _, count = np.unique(component, return_counts=True)
        subset = np.in1d(component, count.argsort()[-1:])
        data=data.subgraph(torch.from_numpy(subset).to(torch.bool))

        ###remove self-loops
        data.edge_index,_=remove_self_loops(data.edge_index)

        ###Determine whether there are duplicate edges
        edges = data.edge_index.t()
        unique_edges, counts = torch.unique(edges, dim=0, return_counts=True)
        duplicate_edges = unique_edges[counts > 1]  
        if duplicate_edges.numel() > 0:
            print(f"There are duplicate edges, the number is {duplicate_edges.shape[0]} ")
            data.edge_index = unique_edges.t()
        else:
            print("No duplicate edges.")

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


class Social(InMemoryDataset):
    url = 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/tree/main/datasets/'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        assert self.name in ['slashdot', 'epinions']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name.lower()}.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url + self.raw_file_names, self.raw_dir)

    def process(self):
        data = []
        node_map = {}
        with open(self.raw_paths[0], 'r') as f:
            for line in f:
                x = line.strip().split(',')
                if float(x[2]) >= 0:
                    assert len(x) == 3
                    a, b = x[0], x[1]
                    if a not in node_map:
                        node_map[a] = len(node_map)
                    
                    if b not in node_map:
                        node_map[b] = len(node_map)
                    
                    a, b = node_map[a], node_map[b]
                    data.append([a, b])

            edge_index = [[i[0], int(i[1])] for i in data]
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = edge_index.t().contiguous()
        
        num_nodes = edge_index.max().item() + 1
        
        #get 100 dimensional features
        #x = set_spectral_adjacency_reg_features(num_nodes, edge_index, torch.ones(edge_index.shape[1]), k=100)
        #x = torch.rand(num_nodes,100)

        #Generate random features
        shape = (num_nodes, 100)
        rnd_state = np.random.RandomState(2025)
        x = torch.tensor(rnd_state.normal(0, 1, shape).astype(np.float32))
        
        data = Data(x=x, edge_index=edge_index)
        ###Select the largest connected components in the graph
        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        num_components, component = sp.csgraph.connected_components(adj)
        _, count = np.unique(component, return_counts=True)
        subset = np.in1d(component, count.argsort()[-1:])
        data=data.subgraph(torch.from_numpy(subset).to(torch.bool))

        ###remove self-loops
        data.edge_index,_=remove_self_loops(data.edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name.capitalize()}()'


