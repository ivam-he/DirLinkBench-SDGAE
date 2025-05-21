import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

#from layers.layers import FermiDiracDecoder
#from layers.layers import GravityDecoder
import manifolds
#import layers.hyp_layers as hyp_layers
import model.dhypr_encoder as encoders
#from models.decoders import model2decoder, SPDecoder
from torch.autograd import Variable
#from utils.eval_utils import acc_f1
#import pdb

class GravityDecoder(Module):
    def __init__(self, manifold, in_features, out_features, c, act, use_bias, beta, lamb):
        super(GravityDecoder, self).__init__()
        self.manifold = manifold
        self.c = c
        
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.beta = beta
        self.lamb = lamb

    def forward(self, x, idx, dist):
        
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        
        h = self.linear.forward(x)   
        mass = self.act(h) 
        
        probs = torch.sigmoid(
            self.beta * (
                mass[idx[:, 1]].view(mass[idx[:, 1]].shape[0])
            ) - self.lamb * (
                torch.log(dist + self.eps[x.dtype])))
        # add eps to avoid nan in probs

        return probs, mass


class FermiDiracDecoder(Module):
    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs


class LPModel(nn.Module):
    def __init__(self, args):
        super(LPModel, self).__init__()
        
        assert args.c is None
        self.c = nn.Parameter(torch.Tensor([1.])) 
        
        self.manifold_name = args.manifold
        self.manifold = getattr(manifolds, self.manifold_name)()
        self.nnodes = args.n_nodes       
        
        self.encoder = getattr(encoders, args.model)(self.c, args)  
        
   
        if not args.act:
            act = lambda x: x
        else:    
            act = getattr(F, args.act)
            
        self.dc = GravityDecoder(
            self.manifold, args.dim, 1, self.c, act, args.bias, args.beta, args.lamb)  
        
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        
        self.fd_dc = FermiDiracDecoder(r=args.r, t=args.t)
    
        
    def encode(self, x, adj):
        h = self.encoder.forward(x, adj)
        return h
    
    
    def decode(self, h, idx): 
        emb_in = h[idx[:, 0], :]   
        emb_out = h[idx[:, 1], :]  
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)  
        # squared distance between pairs of nodes in the hyperbolic space
        probs, mass = self.dc.forward(h, idx, sqdist)
        return probs, mass
    
    
    def fd_decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.fd_dc.forward(sqdist)
        return probs

    def get_loss(self, args, embeddings,  pos_edge, neg_edge):
        if args.wl1 > 0:
            # fermi dirac 
            pos_scores = self.fd_decode(embeddings, pos_edge)
            neg_scores = self.fd_decode(embeddings, neg_edge)
            fd_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
            fd_loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
            
        # gravity 
        pos_scores, mass = self.decode(embeddings, pos_edge)
        neg_scores, mass = self.decode(embeddings, neg_edge)
        g_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        g_loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        
        '''if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        #ap = average_precision_score(labels, preds)'''
        
        if args.wl1 > 0:
            assert args.wl2 > 0
            loss = args.wl1 * fd_loss + args.wl2 * g_loss 
        else:
            loss = g_loss 
      
        #metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return loss

    
    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    
    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])