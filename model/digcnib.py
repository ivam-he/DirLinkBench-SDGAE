from typing import Tuple
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_signed_directed.nn.directed import DiGCNConv
from torch_geometric_signed_directed.utils import get_appr_directed_adj, get_second_directed_adj


class DiGCN_InceptionBlock(torch.nn.Module):
    r"""An implementation of the inception block model from the
    `Digraph Inception Convolutional Networks
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.

    Args:
        in_dim (int): Dimention of input.
        out_dim (int): Dimention of output.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super(DiGCN_InceptionBlock, self).__init__()
        self.ln = nn.Linear(in_dim, out_dim)
        self.conv1 = DiGCNConv(in_dim, out_dim)
        self.conv2 = DiGCNConv(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.ln.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor, edge_index2: torch.LongTensor,
                edge_weight2: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Making a forward pass of the DiGCN inception block model.

        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index, edge_index2 (PyTorch LongTensor) - Edge indices.
            * edge_weight, edge_weight2 (PyTorch FloatTensor) - Edge weights corresponding to edge indices.
        Return types:
            * x0, x1, x2 (PyTorch FloatTensor) - Hidden representations.
        """
        x0 = self.ln(x)
        x1 = self.conv1(x, edge_index, edge_weight)
        x2 = self.conv2(x, edge_index2, edge_weight2)
        return x0, x1, x2


class DiGCNIB(torch.nn.Module):
    r""" This code benefits a lot from PyGSD"""
    def __init__(self, args):
        super(DiGCNIB, self).__init__()
        num_features = args.input_dim
        hidden = args.hidden
        embed_dim = args.embed_dim
        
        self.alpha = args.alpha
        self.ib1 = DiGCN_InceptionBlock(num_features, hidden)
        self.ib2 = DiGCN_InceptionBlock(hidden, hidden)
        self.ib3 = DiGCN_InceptionBlock(hidden, embed_dim)

        self._dropout = args.dropout
        self.reset_parameters()

        ##
        self.dataset = args.dataset

        self.edge_index_tuple, self.edge_weight_tuple = None, None

    def reset_parameters(self):
        self.ib1.reset_parameters()
        self.ib2.reset_parameters()
        self.ib3.reset_parameters()
        #self.linear.reset_parameters()

    def forward(self, x, edge_index_init, split):
        if self.edge_index_tuple is None:
            device = x.device
            cache_dir = './digcnib_cache/'+ self.dataset
            os.makedirs(cache_dir, exist_ok=True) 
            edge_data_path = os.path.join(cache_dir, f'split_{split}_alpha_{self.alpha}.pt')
            time_path = os.path.join(cache_dir, f'split_{split}_alpha_{self.alpha}_time.txt')

            if os.path.exists(edge_data_path):
                cached_data = torch.load(edge_data_path, map_location='cpu')
                edge_index1 = cached_data['edge_index1'].to(device)
                edge_weight1 = cached_data['edge_weight1'].to(device)
                edge_index2 = cached_data['edge_index2'].to(device)
                edge_weight2 = cached_data['edge_weight2'].to(device)
                
            else:
                st_time = time.time()
                edge_index1, edge_weight1 = get_appr_directed_adj(self.alpha, edge_index_init, x.shape[0], x.dtype)
                edge_index2, edge_weight2 = get_second_directed_adj(edge_index_init, x.shape[0], x.dtype)
                
                torch.save({
                    'edge_index1': edge_index1.cpu(),
                    'edge_weight1': edge_weight1.cpu(),
                    'edge_index2': edge_index2.cpu(),
                    'edge_weight2': edge_weight2.cpu()
                }, edge_data_path)

                generation_time = time.time()-st_time
                with open(time_path, 'w') as f:
                    f.write(f"The generation time: {generation_time}\n")
            
            #edge_index1, edge_weight1 = get_appr_directed_adj(self.alpha, edge_index_init, x.shape[0], x.dtype)
            #edge_index2, edge_weight2 = get_second_directed_adj(edge_index_init, x.shape[0], x.dtype)
            
            self.edge_index_tuple = (edge_index1, edge_index2)
            self.edge_weight_tuple = (edge_weight1, edge_weight2)

        #x = features
        edge_index, edge_index2 = self.edge_index_tuple
        edge_weight, edge_weight2 = self.edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight,
                              edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=self._dropout, training=self.training)

        x0, x1, x2 = self.ib2(x, edge_index, edge_weight,
                              edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=self._dropout, training=self.training)

        x0, x1, x2 = self.ib3(x, edge_index, edge_weight,
                              edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2

        #x = torch.cat((x[query_edges[:, 0]], x[query_edges[:, 1]]), dim=-1)
        #x = self.linear(x)
        return x #F.log_softmax(x, dim=1)

