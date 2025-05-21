import numpy as np
import scipy.sparse as sp
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn.init as init

import manifolds
#import layers.hyp_layers as hyp_layers
#from layers.layers import Linear
#import utils.math_utils as pmath
#import pdb
#from utils.data_utils import sparse_mx_to_torch_sparse_tensor, normalize

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class SpecialSpmmFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpAttn(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, activation):
        super(SpAttn, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.act = activation

        
    def forward(self, input, adj, return_attn=False):
        N = input.size()[0]
        edge = adj._indices()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*d x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        
        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones.cuda()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), ones)
        # e_rowsum: N x 1
        
        if return_attn:
            # detach 
            attn_weights = [edge.cpu().detach().numpy(), 
                            edge_e.cpu().detach().numpy(), 
                            e_rowsum.cpu().detach().numpy()]
            
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        
        if return_attn:
            return self.act(h_prime), attn_weights
        else:
            return self.act(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.in_features) + ' -> ' + str(self.out_features) + ')'


def get_dim_act_curv(args):
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.hidden] * (args.num_layers - 1)) 
    dims += [args.dim]
    acts += [act]
    n_curvatures = args.num_layers  
    if args.c is None:
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HypAttnAgg(Module):
    def __init__(self, manifold, c, in_features, dropout, alpha, use_att, n_heads):
        super(HypAttnAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        
        self.nheads = n_heads
        
        if self.use_att:
            if self.nheads > 1:
                assert in_features % self.nheads == 0
                self.att = [SpAttn(
                    in_features, in_features//self.nheads, 
                    dropout, alpha, F.elu) for _ in range(self.nheads)]
                for i, attention in enumerate(self.att):
                    self.add_module('attention_{}'.format(i), attention)
            
            else:
                self.att = SpAttn(in_features, in_features, dropout, alpha, F.elu)
            
    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.nheads > 1:
                support_t_heads = []
                adj_att_heads = []
                for att in self.att:
                    support_t, adj_att = att(x_tangent, adj, return_attn=True)
                    support_t_heads.append(support_t)
                    adj_att_heads.append(adj_att)
                    
                support_t = torch.cat(support_t_heads, dim=1)
            else:
                support_t, adj_att = self.att(x_tangent, adj, return_attn=True)
        else:
            support_t = torch.spmm(adj, x_tangent)
            
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        
        if self.use_att:
            if self.nheads > 1:
                return output, adj_att_heads
            else:
                return output, adj_att
        else:
            return output


    def extra_repr(self):
        return 'c={}'.format(self.c)


class DHYPR(nn.Module):
    def __init__(self, c, args):
        super(DHYPR, self).__init__()
        self.args = args
        
        self.manifold = getattr(manifolds, args.manifold)()
        self.c = c
        
        assert args.num_layers > 1
        self.dims, self.acts, self.curvatures = get_dim_act_curv(args)
        self.curvatures.append(c)  
        
        if args.proximity == 1:
            self.model1_d_i = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model1_d_o = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model1_n_i = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model1_n_o = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
        elif args.proximity == 2:
            self.model1_d_i = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model1_d_o = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model1_n_i = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model1_n_o = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model2_d_i = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model2_d_o = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model2_n_i = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model2_n_o = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
        elif args.proximity == 3:
            self.model1_d_i = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model1_d_o = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model1_n_i = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model1_n_o = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model2_d_i = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model2_d_o = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model2_n_i = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model2_n_o = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model3_d_i = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model3_d_o = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model3_n_i = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
            self.model3_n_o = DHYPRLayer(args, self.manifold, self.dims, self.acts, self.curvatures)
        else:
            os._exit(0)
        
        self.embed_agg = HypAttnAgg(self.manifold, c, self.dims[-1], args.dropout, 
                                               args.alpha, args.use_att, args.n_heads)
        
        self.proximity = args.proximity
        self.nnodes = args.n_nodes
        self.nrepre = args.proximity*4+1
        self.embed_agg_adj_size = args.n_nodes * self.nrepre
        
        embed_agg_adj = np.zeros((self.embed_agg_adj_size, self.embed_agg_adj_size))
        for n in range(self.nnodes):
            block_start = n*self.nrepre
            embed_agg_adj[block_start][block_start+1: block_start+self.nrepre] = 1

        embed_agg_adj = sp.csr_matrix(embed_agg_adj)
        self.embed_agg_adj = sparse_mx_to_torch_sparse_tensor(
            normalize(embed_agg_adj + sp.eye(embed_agg_adj.shape[0]))).to(args.device)
    
        
    def forward(self, x, adj):
        if self.proximity == 1:
            x1_d_is = self.model1_d_i.encode(x, adj['a1_d_i_norm'])
            x1_d_os = self.model1_d_o.encode(x, adj['a1_d_o_norm'])
            x1_n_is = self.model1_n_i.encode(x, adj['a1_n_i_norm'])
            x1_n_os = self.model1_n_o.encode(x, adj['a1_n_o_norm'])
        
            x1_d_i = x1_d_is[-1]
            x1_d_o = x1_d_os[-1]
            x1_n_i = x1_n_is[-1]
            x1_n_o = x1_n_os[-1]
           
            ### target embedding
            target_context = torch.stack((x1_d_i, x1_d_o, x1_n_i, x1_n_o))
            x1_d_i_w = self.manifold.mobius_mulscaler(1.0/8, x1_d_i, self.c)
            x1_d_o_w = self.manifold.mobius_mulscaler(1.0/8, x1_d_o, self.c)
            x1_n_i_w = self.manifold.mobius_mulscaler(1.0/8, x1_n_i, self.c)
            x1_n_o_w = self.manifold.mobius_mulscaler(1.0/8, x1_n_o, self.c)
            
            target = self.manifold.mobius_add(
                self.manifold.mobius_add(
                    self.manifold.mobius_add(x1_d_i_w, x1_d_o_w, self.c), 
                    x1_n_i_w, self.c), 
                x1_n_o_w, self.c)

            target_context_feat = torch.cat((target_context, target.unsqueeze(0)), dim=0).permute(1, 0, 2)
            target_context_feat = target_context_feat.reshape(self.nnodes*self.nrepre, self.dims[-1])

            if self.args.use_att:
                output, output_attn = self.embed_agg.forward(target_context_feat, self.embed_agg_adj)
                output = output.reshape(
                    self.nnodes, self.nrepre, self.dims[-1])[:, 0, :]

                embeddings = [x1_d_is, x1_d_os, x1_n_is, x1_n_os, output]

                return embeddings, output_attn
            else:
                output = self.embed_agg.forward(target_context_feat, self.embed_agg_adj).reshape(
                    self.nnodes, self.nrepre, self.dims[-1])[:, 0, :]

                embeddings = [x1_d_is, x1_d_os, x1_n_is, x1_n_os, output]

                return embeddings
            
        elif self.proximity == 2:
            x1_d_is = self.model1_d_i.encode(x, adj['a1_d_i_norm'])
            x1_d_os = self.model1_d_o.encode(x, adj['a1_d_o_norm'])
            x1_n_is = self.model1_n_i.encode(x, adj['a1_n_i_norm'])
            x1_n_os = self.model1_n_o.encode(x, adj['a1_n_o_norm'])
            x2_d_is = self.model2_d_i.encode(x, adj['a2_d_i_norm'])
            x2_d_os = self.model2_d_o.encode(x, adj['a2_d_o_norm'])
            x2_n_is = self.model2_n_i.encode(x, adj['a2_n_i_norm'])
            x2_n_os = self.model2_n_o.encode(x, adj['a2_n_o_norm'])

            x1_d_i = x1_d_is[-1]
            x1_d_o = x1_d_os[-1]
            x1_n_i = x1_n_is[-1]
            x1_n_o = x1_n_os[-1]
            x2_d_i = x2_d_is[-1]
            x2_d_o = x2_d_os[-1]
            x2_n_i = x2_n_is[-1]
            x2_n_o = x2_n_os[-1]

            ### target embedding
            target_context = torch.stack((x1_d_i, x1_d_o, x1_n_i, x1_n_o, x2_d_i, x2_d_o, x2_n_i, x2_n_o))
            x1_d_i_w = self.manifold.mobius_mulscaler(1.0/8, x1_d_i, self.c)
            x1_d_o_w = self.manifold.mobius_mulscaler(1.0/8, x1_d_o, self.c)
            x1_n_i_w = self.manifold.mobius_mulscaler(1.0/8, x1_n_i, self.c)
            x1_n_o_w = self.manifold.mobius_mulscaler(1.0/8, x1_n_o, self.c)
            x2_d_i_w = self.manifold.mobius_mulscaler(1.0/8, x2_d_i, self.c)
            x2_d_o_w = self.manifold.mobius_mulscaler(1.0/8, x2_d_o, self.c)
            x2_n_i_w = self.manifold.mobius_mulscaler(1.0/8, x2_n_i, self.c)
            x2_n_o_w = self.manifold.mobius_mulscaler(1.0/8, x2_n_o, self.c)

            target = self.manifold.mobius_add(
                self.manifold.mobius_add(
                    self.manifold.mobius_add(
                        self.manifold.mobius_add(
                            self.manifold.mobius_add(
                                self.manifold.mobius_add(
                                    self.manifold.mobius_add(x1_d_i_w, x1_d_o_w, self.c), 
                                    x1_n_i_w, self.c), 
                                x1_n_o_w, self.c), 
                            x2_d_i_w, self.c), 
                        x2_d_o_w, self.c), 
                    x2_n_i_w, self.c), 
                x2_n_o_w, self.c)  

            target_context_feat = torch.cat((target_context, target.unsqueeze(0)), dim=0).permute(1, 0, 2)
            target_context_feat = target_context_feat.reshape(self.nnodes*self.nrepre, self.dims[-1])

            if self.args.use_att:
                output, output_attn = self.embed_agg.forward(target_context_feat, self.embed_agg_adj)
                output = output.reshape(
                    self.nnodes, self.nrepre, self.dims[-1])[:, 0, :]

                embeddings = [x1_d_is, x1_d_os, x1_n_is, x1_n_os, x2_d_is, x2_d_os, x2_n_is, x2_n_os, output]

                return embeddings, output_attn
            else:
                output = self.embed_agg.forward(target_context_feat, self.embed_agg_adj).reshape(
                    self.nnodes, self.nrepre, self.dims[-1])[:, 0, :]

                embeddings = [x1_d_is, x1_d_os, x1_n_is, x1_n_os, x2_d_is, x2_d_os, x2_n_is, x2_n_os, output]

                return embeddings
            
        elif self.proximity == 3:
            x1_d_is = self.model1_d_i.encode(x, adj['a1_d_i_norm'])
            x1_d_os = self.model1_d_o.encode(x, adj['a1_d_o_norm'])
            x1_n_is = self.model1_n_i.encode(x, adj['a1_n_i_norm'])
            x1_n_os = self.model1_n_o.encode(x, adj['a1_n_o_norm'])
            x2_d_is = self.model2_d_i.encode(x, adj['a2_d_i_norm'])
            x2_d_os = self.model2_d_o.encode(x, adj['a2_d_o_norm'])
            x2_n_is = self.model2_n_i.encode(x, adj['a2_n_i_norm'])
            x2_n_os = self.model2_n_o.encode(x, adj['a2_n_o_norm'])
            x3_d_is = self.model3_d_i.encode(x, adj['a3_d_i_norm'])
            x3_d_os = self.model3_d_o.encode(x, adj['a3_d_o_norm'])
            x3_n_is = self.model3_n_i.encode(x, adj['a3_n_i_norm'])
            x3_n_os = self.model3_n_o.encode(x, adj['a3_n_o_norm'])

            x1_d_i = x1_d_is[-1]
            x1_d_o = x1_d_os[-1]
            x1_n_i = x1_n_is[-1]
            x1_n_o = x1_n_os[-1]
            x2_d_i = x2_d_is[-1]
            x2_d_o = x2_d_os[-1]
            x2_n_i = x2_n_is[-1]
            x2_n_o = x2_n_os[-1]
            x3_d_i = x3_d_is[-1]
            x3_d_o = x3_d_os[-1]
            x3_n_i = x3_n_is[-1]
            x3_n_o = x3_n_os[-1]

            ### target embedding
            target_context = torch.stack((x1_d_i, x1_d_o, x1_n_i, x1_n_o, 
                                          x2_d_i, x2_d_o, x2_n_i, x2_n_o, 
                                          x3_d_i, x3_d_o, x3_n_i, x3_n_o))
            x1_d_i_w = self.manifold.mobius_mulscaler(1.0/8, x1_d_i, self.c)
            x1_d_o_w = self.manifold.mobius_mulscaler(1.0/8, x1_d_o, self.c)
            x1_n_i_w = self.manifold.mobius_mulscaler(1.0/8, x1_n_i, self.c)
            x1_n_o_w = self.manifold.mobius_mulscaler(1.0/8, x1_n_o, self.c)
            x2_d_i_w = self.manifold.mobius_mulscaler(1.0/8, x2_d_i, self.c)
            x2_d_o_w = self.manifold.mobius_mulscaler(1.0/8, x2_d_o, self.c)
            x2_n_i_w = self.manifold.mobius_mulscaler(1.0/8, x2_n_i, self.c)
            x2_n_o_w = self.manifold.mobius_mulscaler(1.0/8, x2_n_o, self.c)
            x3_d_i_w = self.manifold.mobius_mulscaler(1.0/8, x3_d_i, self.c)
            x3_d_o_w = self.manifold.mobius_mulscaler(1.0/8, x3_d_o, self.c)
            x3_n_i_w = self.manifold.mobius_mulscaler(1.0/8, x3_n_i, self.c)
            x3_n_o_w = self.manifold.mobius_mulscaler(1.0/8, x3_n_o, self.c)

            target = self.manifold.mobius_add(
                self.manifold.mobius_add(
                    self.manifold.mobius_add(
                        self.manifold.mobius_add(
                            self.manifold.mobius_add(
                                self.manifold.mobius_add(
                                    self.manifold.mobius_add(
                                        self.manifold.mobius_add(
                                            self.manifold.mobius_add(
                                                self.manifold.mobius_add(
                                                    self.manifold.mobius_add(x1_d_i_w, x1_d_o_w, self.c), 
                                                    x1_n_i_w, self.c), 
                                                x1_n_o_w, self.c), 
                                            x2_d_i_w, self.c), 
                                        x2_d_o_w, self.c), 
                                    x2_n_i_w, self.c), 
                                x2_n_o_w, self.c), 
                            x3_d_i_w, self.c), 
                        x3_d_o_w, self.c), 
                    x3_n_i_w, self.c), 
                x3_n_o_w, self.c)
                                                                                                                         
            target_context_feat = torch.cat((target_context, target.unsqueeze(0)), dim=0).permute(1, 0, 2)
            target_context_feat = target_context_feat.reshape(self.nnodes*self.nrepre, self.dims[-1])

            if self.args.use_att:
                output, output_attn = self.embed_agg.forward(target_context_feat, self.embed_agg_adj)
                output = output.reshape(
                    self.nnodes, self.nrepre, self.dims[-1])[:, 0, :]

                embeddings = [x1_d_is, x1_d_os, x1_n_is, x1_n_os, 
                              x2_d_is, x2_d_os, x2_n_is, x2_n_os, 
                              x3_d_is, x3_d_os, x3_n_is, x3_n_os, output]

                return embeddings, output_attn
            else:
                output = self.embed_agg.forward(target_context_feat, self.embed_agg_adj).reshape(
                    self.nnodes, self.nrepre, self.dims[-1])[:, 0, :]

                embeddings = [x1_d_is, x1_d_os, x1_n_is, x1_n_os, 
                              x2_d_is, x2_d_os, x2_n_is, x2_n_os,
                              x3_d_is, x3_d_os, x3_n_is, x3_n_os, output]

                return embeddings
        
        

class HyperbolicGraphConvolution(nn.Module):
    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)   
        h = self.agg.forward(h, adj)  
        h = self.hyp_act.forward(h)  
        return h

class HypLinear(nn.Module):
    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias: 
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res
        
    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
                self.in_features, self.out_features, self.c
        )

class HypAgg(Module):
    def __init__(self, manifold, c, in_features, dropout):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)   
        support_t = torch.spmm(adj, x_tangent)  
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)   
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)
    

class HypAct(Module):
    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x): 
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
                self.c_in, self.c_out
        )


class DHYPRLayer(nn.Module):
    def __init__(self, args, manifold, dims, acts, curvatures):
        super(DHYPRLayer, self).__init__()
        self.manifold = manifold
        self.curvatures = curvatures
        hgc_layers = []
        for i in range(len(dims) - 1): 
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                        HyperbolicGraphConvolution(
                        self.manifold, in_dim, out_dim, c_in, c_out, 
                        args.dropout, act, args.bias
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])  
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])   
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])   
        
        embeddings = []
        input = (x_hyp, adj)
        for layer in self.layers:
            x_hyp = layer.forward(input)
            input = (x_hyp, adj)
            embeddings.append(x_hyp)
            
        return embeddings
