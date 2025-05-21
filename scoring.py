import torch
import torch.nn.functional as F
import torch.nn as nn
# from torch_sparse.matmul import  spmm_add

class InnerProduct(torch.nn.Module):
    """Inner product"""
    def forward(self, x_i, x_j):
        value = (x_i * x_j).sum(dim=1)
        return torch.sigmoid(value)

class MLPScore(torch.nn.Module):
    def __init__(self, args):
        super(MLPScore, self).__init__()
        hidden_channels = args.hidden_predictor
        out_channels = args.outdim_predictor
        num_layers = args.nlayer_predictor
        self.operator = args.operator
        if self.operator == "cat":
            if args.net == "DGCN":
                in_channels = args.embed_dim*6
            else:
                in_channels = args.embed_dim*2

        elif self.operator == "hadamard":
            if args.net == "DGCN":
                in_channels = args.embed_dim*3
            else:
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

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        if self.operator == "cat":
            x = torch.cat((x_i, x_j), dim=-1) #concat
        elif self.operator == "hadamard":
            x = x_i * x_j #hadamard

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class SingleClassifier(torch.nn.Module):
    """Linear classifiers, widely used in previous benchmarks for the single embedding methods."""
    def __init__(self, args):
        super(SingleClassifier, self).__init__()
        in_channels = args.embed_dim*2
        out_channels = args.outdim_predictor

        if args.net == "DGCN":
            in_channels = args.embed_dim*6
        
        self.dropout = args.dropout
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = torch.cat((x_i, x_j), dim=-1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

class DualClassifier(torch.nn.Module):
    """Linear classifiers, widely used in previous benchmarks for the Dual embedding methods, like MagNet."""
    def __init__(self, args):
        super(DualClassifier, self).__init__()
        in_channels = args.embed_dim*4
        out_channels = args.outdim_predictor

        self.lin = nn.Linear(in_channels, out_channels)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, s_i, s_j, t_i, t_j):
        x = torch.cat((s_i, s_j, t_i, t_j), dim=-1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


class DualMLPScore(torch.nn.Module):
    def __init__(self, args):
        super(DualMLPScore, self).__init__()
        hidden_channels = args.hidden_predictor
        out_channels = args.outdim_predictor
        num_layers = args.nlayer_predictor

        in_channels = args.embed_dim*4
        
        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = args.dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, s_i, s_j, t_i, t_j):
        x = torch.cat((s_i, s_j, t_i, t_j), dim=-1)

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
