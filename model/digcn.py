import torch
import os
import time
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_signed_directed.nn.directed import DiGCNConv
from torch_geometric_signed_directed.utils import get_appr_directed_adj


class DiGCN(torch.nn.Module):
    r""" This code benefits a lot from PyGSD"""
    def __init__(self, args):
        super(DiGCN, self).__init__()
        num_features = args.input_dim
        hidden = args.hidden
        embed_dim = args.embed_dim
        self.alpha = args.alpha
        
        self.conv1 = DiGCNConv(num_features, hidden)
        self.conv2 = DiGCNConv(hidden, embed_dim)
        self.dropout = args.dropout

        self.reset_parameters()

        ##
        self.dataset = args.dataset

        self.edge_index, self.edge_weight = None, None

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        #self.linear.reset_parameters()

    def forward(self, x, edge_index, split):
        if self.edge_index is None:
            device = x.device
            cache_dir = './digcn_cache/'+ self.dataset
            os.makedirs(cache_dir, exist_ok=True) 
            edge_data_path = os.path.join(cache_dir, f'split_{split}_alpha_{self.alpha}.pt')
            time_path = os.path.join(cache_dir, f'split_{split}_alpha_{self.alpha}_time.txt')

            if os.path.exists(edge_data_path):
                cached_data = torch.load(edge_data_path, map_location='cpu')
                self.edge_index = cached_data['edge_index'].to(device)
                self.edge_weight = cached_data['edge_weight'].to(device)
                
            else:
                st_time = time.time()
                self.edge_index, self.edge_weight = get_appr_directed_adj(self.alpha, edge_index, x.shape[0], x.dtype)
                torch.save({
                    'edge_index': self.edge_index.cpu(),
                    'edge_weight': self.edge_weight.cpu()
                }, edge_data_path)

                generation_time = time.time()-st_time
                with open(time_path, 'w') as f:
                    f.write(f"The generation time: {generation_time}\n")

            #self.edge_index, self.edge_weight = get_appr_directed_adj(self.alpha, edge_index, x.shape[0], x.dtype)
        
        x = F.relu(self.conv1(x, self.edge_index, self.edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, self.edge_index, self.edge_weight)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        #x = torch.cat((x[query_edges[:, 0]], x[query_edges[:, 1]]), dim=-1)
        #x = self.linear(x)
        return x #F.log_softmax(x, dim=1)