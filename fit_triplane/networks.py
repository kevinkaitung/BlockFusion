import os, pdb, sys, pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import distributions as dist

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

# https://github.com/yenchenlin/nerf-pytorch/blob/63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd/run_nerf_helpers.py#L48
# copy from HyperDiffusion
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)
        # 1 0 yap
        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class HyperDiffusionMLP(nn.Module):
    def __init__(
        self,
        out_size,
        hidden_neurons,
        use_leaky_relu=False,
        use_bias=True,
        multires=10,
        output_type=None,
        move=False,
        **kwargs,
    ):
        super().__init__()
        self.embedder = Embedder(
            include_input=True,
            input_dims=3 if not move else 4,
            max_freq_log2=multires - 1,
            num_freqs=multires,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.layers = nn.ModuleList([])
        self.output_type = output_type
        self.use_leaky_relu = use_leaky_relu
        in_size = self.embedder.out_dim
        self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
            )
        self.layers.append(nn.Linear(hidden_neurons[-1], out_size, bias=use_bias))

    def forward(self, model_input):
        # NOTE: I don't need to keep input coords
        # coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        # x = coords_org
        x = model_input
        x = self.embedder.embed(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.leaky_relu(x) if self.use_leaky_relu else F.relu(x)
        x = self.layers[-1](x)

        # if self.output_type == "occ":
        #     # x = torch.sigmoid(x)
        #     pass
        # elif self.output_type == "sdf":
        #     x = torch.tanh(x)
        # elif self.output_type == "logits":
        #     x = x
        # else:
        #     raise f"This self.output_type ({self.output_type}) not implemented"
        # x = dist.Bernoulli(logits=x).logits

        # NOTE: I don't need to return original coords
        # return {"model_in": coords_org, "model_out": x}
        return x
