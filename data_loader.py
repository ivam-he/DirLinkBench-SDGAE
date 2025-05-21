import torch
import math
import pickle
import os.path as osp
import os
import time
import numpy as np
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import from_scipy_sparse_matrix,to_scipy_sparse_matrix,remove_self_loops, is_undirected, to_networkx
import scipy.sparse as sp
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from dataset import Citation, Amazon, WikiCS, Social
from utils import GraphInfo

def dataloader(name):
    root = "./data"
    name = name.lower()
    if name in ['cora_ml', 'citeseer']:
        dataset = Citation(root, name, transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        path = osp.join(root, name)
        dataset = Amazon(path, name, transform=T.NormalizeFeatures())
    elif name == "wikics":
        path = osp.join(root,name)
        dataset = WikiCS(path,transform=T.NormalizeFeatures())
    elif name in ["slashdot", "epinions"]:
        path = osp.join(root,name)
        dataset = Social(path, name)
    print("==============================")
    print(f"The information of {name} is:")
    GraphInfo(dataset[0])
    return dataset




