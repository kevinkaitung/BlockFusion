import os, pdb, sys, pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchist
import numpy as np

# copy from https://github.com/wilsonCernWq/instant-vnr-pytorch/blob/main/core/networks.py

class SineLayer(nn.Module):
    '''Reference: https://github.com/matthewberger/neurcomp/blob/main/siren.py'''
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super(SineLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.omega_0 = omega_0
        self.is_first = is_first

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, inputs):
        return torch.sin(self.omega_0 * self.linear(inputs))


class SirenResBlock(nn.Module):
    '''Reference: https://github.com/matthewberger/neurcomp/blob/main/siren.py'''
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super(SirenResBlock, self).__init__()

        self.features = features
        self.omega_0 = omega_0

        self.linear_1 = nn.Linear(features, features, bias=bias)
        self.linear_2 = nn.Linear(features, features, bias=bias)

        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)

    def forward(self, inputs):
        sine_1 = torch.sin(self.omega_0 * self.linear_1(self.weight_1 * inputs))
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2 * (inputs + sine_2)

class NeurCompNet(torch.nn.Module):
    def __init__(self, n_input_dims=3, n_output_dims=1, bias=False, n_hidden_layers=8, n_neurons=256, is_residual=True):
        super(NeurCompNet, self).__init__()

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims

        self.n_hidden_layers = n_hidden_layers
        self.n_layers = n_hidden_layers + 2
        self.n_neurons = n_neurons
        self.bias = bias
        self.is_residual = is_residual

        net = []
        for l in range(self.n_layers):
            in_dim  = self.n_input_dims  if l == 0 else self.n_neurons
            out_dim = self.n_output_dims if l == self.n_layers - 1 else self.n_neurons
            is_first = (l==0)
            if l != self.n_layers-1:
                if not self.is_residual:
                    net.append(SineLayer(in_dim, out_dim, bias=True, is_first=is_first))
                else:
                    if is_first:
                        net.append(SineLayer(in_dim, out_dim, bias=True, is_first=is_first))
                    else:
                        net.append(SirenResBlock(in_dim, bias=True, ave_first=(l>1), ave_second=(l==(self.n_layers-2))))
            else:
                final_linear = nn.Linear(in_dim, out_dim)
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6 / (in_dim)) / 30.0, np.sqrt(6 / (in_dim)) / 30.0)
                net.append(final_linear)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        *S, C = x.size()
        assert C == self.n_input_dims
        x = x.view(-1, self.n_input_dims) * 2 - 1     # to [-1, 1]
        x = self.net(x) * 0.5 + 0.5                   # to [ 0, 1]
        return x.view(*S, self.n_output_dims)