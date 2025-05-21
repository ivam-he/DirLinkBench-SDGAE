import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import negative_sampling, remove_self_loops
from sklearn.metrics import roc_auc_score, average_precision_score
#from layers import DirectedInnerProductDecoder
#from initializations import reset

from model.sdgae import SimpleDecoder

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


EPS        = 1e-15
MAX_LOGSTD = 10

################################################################################
# DECODER for DIRECTED models
################################################################################
class DirectedInnerProductDecoder(torch.nn.Module):
    def forward(self, s, t, edge_index, sigmoid=True):
        value = (s[edge_index[0]] * t[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, s, t, sigmoid=True):
        adj = torch.matmul(s, t.t())
        return torch.sigmoid(adj) if sigmoid else adj

################################################################################
# DIRECTED model layers: alpha, beta are supplied, BASIC version
################################################################################
class DirectedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(DirectedGCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

        # if adaptive is True:
        #     self.alpha = torch.nn.Parameter(torch.Tensor([alpha]))
        #     self.beta  = torch.nn.Parameter(torch.Tensor([beta]))
        # else:
        #     self.alpha      = alpha
        #     self.beta       = beta

        self.alpha      = alpha
        self.beta       = beta

        self.self_loops = self_loops
        self.adaptive   = adaptive

    
    def forward(self, x, edge_index):
        if self.self_loops is True:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col  = edge_index

        in_degree  = degree(col)
        out_degree = degree(row)

        alpha = self.alpha
        beta  = self.beta 

        in_norm_inv  = pow(in_degree,  -alpha)
        out_norm_inv = pow(out_degree, -beta)

        in_norm  = in_norm_inv[col]
        out_norm = out_norm_inv[row]
        norm     = in_norm * out_norm

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

################################################################################
# DIRECTED models: two layer
################################################################################
class SourceGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(SourceGCNConvEncoder, self).__init__()
        self.conv1 = DirectedGCNConv(in_channels, hidden_channels, alpha, beta, self_loops, adaptive)
        self.conv2 = DirectedGCNConv(hidden_channels, out_channels, alpha, beta, self_loops, adaptive)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        # x = self.conv1(x, edge_index)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, torch.flip(edge_index, [0]))
        return x

class TargetGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(TargetGCNConvEncoder, self).__init__()
        self.conv1 = DirectedGCNConv(in_channels, hidden_channels, alpha, beta, self_loops, adaptive)
        self.conv2 = DirectedGCNConv(hidden_channels, out_channels, alpha, beta, self_loops, adaptive)
        

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x, torch.flip(edge_index, [0])))
        # x = self.conv1(x, torch.flip(edge_index, [0]))

        # x = F.dropout(x, p=0.5, training=self.training) 
        x = self.conv2(x, edge_index)

        return x

class DirectedGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(DirectedGCNConvEncoder, self).__init__()
        self.source_conv = SourceGCNConvEncoder(in_channels, hidden_channels, out_channels, alpha, beta, self_loops, adaptive)
        self.target_conv = TargetGCNConvEncoder(in_channels, hidden_channels, out_channels, alpha, beta, self_loops, adaptive)

    def forward(self, s, t, edge_index):
        s = self.source_conv(s, edge_index)
        t = self.target_conv(t, edge_index)
        return s, t


################################################################################
# DIRECTED models: single layer
################################################################################
class SingleLayerSourceGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(SingleLayerSourceGCNConvEncoder, self).__init__()
        self.conv = DirectedGCNConv(in_channels, out_channels, alpha, beta, self_loops, adaptive)

    def forward(self, x, edge_index):
        # x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv(x, torch.flip(edge_index, [0]))

        return x

class SingleLayerTargetGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(SingleLayerTargetGCNConvEncoder, self).__init__()
        self.conv = DirectedGCNConv(in_channels, out_channels, alpha, beta, self_loops, adaptive)
        
    def forward(self, x, edge_index):
        # x = F.relu(self.conv1(x, torch.flip(edge_index, [0])))
        # x = F.dropout(x, p=0.5, training=self.training) 
        x = self.conv(x, edge_index)

        return x
    
class SingleLayerDirectedGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(SingleLayerDirectedGCNConvEncoder, self).__init__()
        self.source_conv = SingleLayerSourceGCNConvEncoder(in_channels, out_channels, alpha, beta, self_loops, adaptive)
        self.target_conv = SingleLayerTargetGCNConvEncoder(in_channels, out_channels, alpha, beta, self_loops, adaptive)

    def forward(self, s_0, t_0, edge_index):
        s_1 = self.source_conv(t_0, edge_index)
        t_1 = self.target_conv(s_0, edge_index)
        return s_1, t_1


class DiGAE(torch.nn.Module):
    def __init__(self, args):
        super(DiGAE, self).__init__()
        in_channels = args.input_dim
        hidden_channels = args.hidden
        out_channels = args.embed_dim

        if args.single_layer:
            self.encoder = SingleLayerDirectedGCNConvEncoder(in_channels, out_channels, alpha=args.alpha, beta=args.beta, self_loops=args.self_loops, adaptive=args.adaptive)
        else:   
            self.encoder = DirectedGCNConvEncoder(in_channels, hidden_channels, out_channels, alpha=args.alpha, beta=args.beta, self_loops=args.self_loops, adaptive=args.adaptive)
        
        if args.decoder == "mlpscore":
            self.decoder = SimpleDecoder(args)

        else:
            self.decoder = DirectedInnerProductDecoder()

        DiGAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        s, t = self.encoder(x, x, edge_index)
        adj_pred = self.decoder.forward_all(s, t)
        return adj_pred

    
    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    
    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    
    def recon_loss(self, s, t, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(self.decoder(s, t, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, s.size(0))
        neg_loss = -torch.log(1 - self.decoder(s, t, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    def get_loss(self, s, t, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(self.decoder(s, t, pos_edge_index) + EPS).mean()
        neg_loss = -torch.log(1 - self.decoder(s, t, neg_edge_index) + EPS).mean()

        return pos_loss + neg_loss

    def test(self, s, t, pos_edge_index, neg_edge_index):
        # XXX
        pos_y = s.new_ones(pos_edge_index.size(1))
        neg_y = s.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(s, t, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(s, t, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

