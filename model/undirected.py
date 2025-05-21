import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import APPNP as APPNPConv



class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.lin1 = Linear(args.input_dim, args.hidden)
        self.lin2 = Linear(args.hidden, args.embed_dim)

        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))   
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        #x = F.relu(x)

        return x 

class GCN(torch.nn.Module):
    r"""GCN for Link Prediction"""
    def __init__(self, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(args.input_dim, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.embed_dim)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x


class GAT(torch.nn.Module):
    r"""GAT for Link Prediction"""
    def __init__(self, args):
        super(GAT, self).__init__()
        self.conv1 = GATConv(args.input_dim, args.hidden, heads=args.heads, dropout=args.dropout)
        self.conv2 = GATConv(args.hidden * args.heads, args.embed_dim, heads=args.output_heads, dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        return x

class APPNP(torch.nn.Module):
    r"""APPNP for Link Prediction"""
    def __init__(self, args):
        super(APPNP, self).__init__()
        self.lin1 = Linear(args.input_dim, args.hidden)
        self.lin2 = Linear(args.hidden, args.embed_dim)
        self.prop1 = APPNPConv(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))   
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)


        x = self.prop1(x, edge_index)

        return x


class GPRGNN(torch.nn.Module):
    r"""GPRGNN for Link Prediction"""
    def __init__(self, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(args.input_dim, args.hidden)
        self.lin2 = Linear(args.hidden, args.embed_dim)
        self.prop1 = GPR_prop(args.K, args.alpha, "PPR")
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))   
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)


        x = self.prop1(x, edge_index)

        return x


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''
    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype, add_self_loops=False)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)