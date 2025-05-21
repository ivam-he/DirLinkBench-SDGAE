from typing import Optional
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_signed_directed.nn.directed import DGCNConv
from torch_geometric_signed_directed.utils import directed_features_in_out

class DGCN(torch.nn.Module):
    r""" This code benefits a lot from PyGSD"""
    def __init__(self, args):
        super(DGCN, self).__init__()
        num_features = args.input_dim
        hidden = args.hidden
        embed_dim = args.embed_dim
        
        improved = False
        cached = True
        self.dropout = args.dropout
        self.dgconv = DGCNConv(improved=improved, cached=cached)
        #self.linear = nn.Linear(hidden*6, label_dim)

        self.lin1 = torch.nn.Linear(num_features, hidden, bias=False)
        self.lin2 = torch.nn.Linear(hidden*3, embed_dim, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, hidden))
        self.bias2 = nn.Parameter(torch.Tensor(1, embed_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        ##
        self.dataset = args.dataset

        self.edge_index, self.edge_in, self.in_w, self.edge_out, self.out_w = None,None,None,None,None

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)
        #self.linear.reset_parameters()

    def forward(self, x, edge_index, split):
        if self.edge_index is None:
            device = x.device
            cache_dir = './dgcn_cache/'+ self.dataset
            os.makedirs(cache_dir, exist_ok=True) 
            edge_data_path = os.path.join(cache_dir, f'split_{split}.pt')
            time_path = os.path.join(cache_dir, f'split_{split}_time.txt')

            if os.path.exists(edge_data_path):
                cached_data = torch.load(edge_data_path, map_location='cpu')
                self.edge_index = cached_data['edge_index'].to(device)
                self.edge_in = cached_data['edge_in'].to(device)
                self.in_w = cached_data['in_w'].to(device)
                self.edge_out = cached_data['edge_out'].to(device)
                self.out_w = cached_data['out_w'].to(device)
            else:
                st_time = time.time()
                self.edge_index, self.edge_in, self.in_w, self.edge_out, self.out_w = directed_features_in_out(edge_index, len(x))
                torch.save({
                    'edge_index': self.edge_index.cpu(),
                    'edge_in': self.edge_in.cpu(),
                    'in_w': self.in_w.cpu(),
                    'edge_out': self.edge_out.cpu(),
                    'out_w': self.out_w.cpu()
                }, edge_data_path)

                generation_time = time.time()-st_time
                with open(time_path, 'w') as f:
                    f.write(f"The generation time: {generation_time}\n")

            #self.edge_index, self.edge_in, self.in_w, self.edge_out, self.out_w = directed_features_in_out(edge_index, len(x))     
        
        x = self.lin1(x)
        x1 = self.dgconv(x, self.edge_index)
        x2 = self.dgconv(x, self.edge_in, self.in_w)
        x3 = self.dgconv(x, self.edge_out, self.out_w)

        x1 += self.bias1
        x2 += self.bias1
        x3 += self.bias1

        x = torch.cat((x1, x2, x3), axis=-1)
        x = F.relu(x)

        x = self.lin2(x)
        x1 = self.dgconv(x, self.edge_index)
        x2 = self.dgconv(x, self.edge_in, self.in_w)
        x3 = self.dgconv(x, self.edge_out, self.out_w)

        x1 += self.bias2
        x2 += self.bias2
        x3 += self.bias2

        x = torch.cat((x1, x2, x3), axis=-1)
        x = F.relu(x)
        

        #x = torch.cat((x[query_edges[:, 0]], x[query_edges[:, 1]]), dim=-1)

        #if self.dropout > 0:
        #    x = F.dropout(x, self.dropout, training=self.training)
        #x = self.linear(x)
        return x #F.log_softmax(x, dim=1)