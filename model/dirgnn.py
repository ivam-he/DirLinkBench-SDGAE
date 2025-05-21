import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import JumpingKnowledge
from torch_sparse import SparseTensor, mul
from torch_sparse import sum as sparsesum
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor 


def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)

    return mul(adj, 1 / row_sum.view(-1, 1))


def directed_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
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

def gcn_que_norm(adj):
    in_deg = sparsesum(adj, dim=0)
    out_deg = sparsesum(adj, dim=1)

    deg = in_deg+out_deg
    hhh =  torch.ones(deg.size(0)).to(0)
    deg = deg + hhh
    #print(deg)
    #print(torch.ones(deg.size(0)))

    #breakpoint()
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, deg_inv_sqrt.view(1, -1))
    return adj


def get_norm_adj(adj, norm):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)
    elif norm == "gcn":
        return gcn_que_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")


'''The implementation of DirGNN from the original paper.'''
class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        return self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(self.adj_t_norm @ x)


class DirGNN(torch.nn.Module):
    def __init__(self,args):
        super(DirGNN, self).__init__()
        num_features = args.input_dim
        hidden_dim = args.hidden
        embed_dim = args.embed_dim
        
        alpha = args.alpha
        num_layers = args.nlayer
        learned_alpha = False
        jumping_knowledge = None
        if args.jk in ['cat','max']:
            jumping_knowledge = args.jk

        self.alpha = torch.nn.Parameter(torch.ones(1) * alpha, requires_grad=learned_alpha)

        if num_layers == 1:
            self.convs = torch.nn.ModuleList([DirGCNConv(num_features, embed_dim, self.alpha)])
        else:
            self.convs = torch.nn.ModuleList([DirGCNConv(num_features, hidden_dim, self.alpha)])
            for _ in range(num_layers - 2):
                self.convs.append(DirGCNConv(hidden_dim, hidden_dim, self.alpha))
            self.convs.append(DirGCNConv(hidden_dim, embed_dim, self.alpha))

        if jumping_knowledge is not None:
            input_dim = hidden_dim * num_layers if jumping_knowledge == "cat" else embed_dim
            self.lin = Linear(input_dim, embed_dim)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = args.dropout
        self.jumping_knowledge = jumping_knowledge
        
        self.normalize = args.normalize

    def forward(self, x, edge_index, edge_weight=None):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jumping_knowledge is not None:
            x = self.jump(xs)
            x = self.lin(x)

        #x = torch.cat((x[index[:,0]], x[index[:,1]]), dim=-1)
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = self.linear(x)

        return x #F.log_softmax(x, dim=1)


