import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import JumpingKnowledge
from torch_sparse import SparseTensor, mul
from torch_sparse import sum as sparsesum
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor 
from torch_geometric.utils import add_self_loops

EPS = 1e-15

def directed_norm(adj):
    """
    Applies the normalization for directed graphs:
        {D}_{out}^{-1/2} {A} {D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj



class SimpleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleConv, self).__init__()

        TEMP = np.ones(2)
        self.weights = torch.nn.Parameter(torch.tensor(TEMP))

        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, s, t, edge_index):
        if self.adj_norm is None:
            num_nodes = s.shape[0]
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            row, col = edge_index
            
            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = directed_norm(adj)
            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = directed_norm(adj_t)

        s_res = s + self.weights[0]*(self.adj_norm @ t)
        t_res = t + self.weights[1]*(self.adj_t_norm @ s)

        return s_res, t_res

class SimpleEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nlayers, dropout, K, Init="ONE"):
        super(SimpleEncoder, self).__init__()

        if nlayers == 1:
            self.lins = torch.nn.ModuleList([Linear(in_channels, out_channels)])
            self.lint = torch.nn.ModuleList([Linear(in_channels, out_channels)])
        else:
            self.lins = torch.nn.ModuleList([Linear(in_channels, hidden_channels)])
            self.lint = torch.nn.ModuleList([Linear(in_channels, hidden_channels)])
            for _ in range(nlayers - 2):
                self.lins.append(Linear(hidden_channels, hidden_channels))
                self.lint.append(Linear(hidden_channels, hidden_channels))
            self.lins.append(Linear(hidden_channels, out_channels))
            self.lint.append(Linear(hidden_channels, out_channels))

        if K == 1:
            self.convs = torch.nn.ModuleList([SimpleConv(in_channels, out_channels)])
        else:
            self.convs = torch.nn.ModuleList([SimpleConv(in_channels, hidden_channels)])
            for _ in range(K - 2):
                self.convs.append(SimpleConv(hidden_channels, hidden_channels))
            self.convs.append(SimpleConv(hidden_channels, out_channels))

        #self.prop = SimpleProp(K, Init)
        self.num_layers = nlayers
        self.dropout = dropout
     

    def forward(self, s, t, edge_index):
        
        for i, lin in enumerate(self.lins):
            s = lin(s)
            if i != self.num_layers - 1:
                s = F.relu(s)
                s = F.dropout(s, p=self.dropout, training=self.training)

        for i, lin in enumerate(self.lint):
            t = lin(t)
            if i != self.num_layers - 1:
                t = F.relu(t)
                t = F.dropout(t, p=self.dropout, training=self.training)
        
        for i, conv in enumerate(self.convs):
            s, t = conv(s, t, edge_index)

        return s, t


class SimpleDecoder(torch.nn.Module):
    def __init__(self, args):
        super(SimpleDecoder, self).__init__()
        self.decoder = args.decoder
        if self.decoder == "mlpscore":
            hidden_channels = args.hidden_predictor
            out_channels = args.outdim_predictor
            num_layers = args.nlayer_predictor
            self.operator = args.operator
            
            if self.operator == "cat":
                in_channels = args.embed_dim*2
            elif self.operator == "hadamard":
                in_channels = args.embed_dim

            self.lins = torch.nn.ModuleList()
            if num_layers == 1: 
                self.lins.append(torch.nn.Linear(in_channels, out_channels))
            else:
                self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
                self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

            self.dropout = args.dropout

        else:
            pass #inner product


    def forward(self, s, t, index):

        if self.decoder == "mlpscore":
            if self.operator == "cat":
                x = torch.cat((s[index[0]], t[index[1]]), dim=-1) #concat
            elif self.operator == "hadamard":
                x = s[index[0]] * t[index[1]] #hadamard

            for lin in self.lins[:-1]:
                x = lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
            return torch.sigmoid(x)
        
        else:
            value = (s[index[0]] * t[index[1]]).sum(dim=1)
            return torch.sigmoid(value)


class SDGAE(torch.nn.Module):
    def __init__(self, args):
        super(SDGAE, self).__init__()
        in_channels = args.input_dim
        hidden_channels = args.hidden
        out_channels = args.embed_dim
        nlayers = args.nlayer
        dropout = args.dropout
        K = args.K
        Init = args.Init

        self.encoder = SimpleEncoder(in_channels, hidden_channels, out_channels, nlayers, dropout, K, Init) 
        self.decoder = SimpleDecoder(args)    
        
    
    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    
    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    
    def get_loss(self, s, t, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(self.decoder(s, t, pos_edge_index) + EPS).mean()
        neg_loss = -torch.log(1 - self.decoder(s, t, neg_edge_index) + EPS).mean()

        return pos_loss + neg_loss

