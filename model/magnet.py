from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_signed_directed.nn.directed import MagNetConv, complex_relu_layer


class MagNet(torch.nn.Module):
    r""" This code benefits a lot from PyGSD"""
    def __init__(self, args):
        super(MagNet, self).__init__()

        num_features = args.input_dim
        hidden = args.hidden
        embed_dim = args.embed_dim
        q = args.q
        K = args.K
        layer = args.nlayer
        dropout = args.dropout
        trainable_q = False
        cached = True
        
        self.normalization = 'sym' 
        self.activation = True
        self.Chebs = nn.ModuleList()
        if layer == 1:
            self.Chebs.append(MagNetConv(in_channels=num_features, out_channels=embed_dim, K=K, q=q, trainable_q=trainable_q, normalization=self.normalization, cached=cached))
        else:
            self.Chebs.append(MagNetConv(in_channels=num_features, out_channels=hidden, K=K, q=q, trainable_q=trainable_q, normalization=self.normalization, cached=cached))
            for _ in range(layer-2):
                self.Chebs.append(MagNetConv(in_channels=hidden, out_channels=hidden, K=K, q=q, trainable_q=trainable_q, normalization=self.normalization, cached=cached))
            self.Chebs.append(MagNetConv(in_channels=hidden, out_channels=embed_dim, K=K, q=q, trainable_q=trainable_q, normalization=self.normalization, cached=cached))

        if self.activation:
            self.complex_relu = complex_relu_layer()
        self.dropout = dropout

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        real = x
        imag = x.clone()
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)
                
        return real, imag

