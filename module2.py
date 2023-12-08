# -*- coding: utf-8 -*-

from typing import Callable, Optional

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    
    def __init__(self, n_in, n_out):
        super(GraphConvolution, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.weight = Parameter(torch.FloatTensor(n_in, n_out))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
            
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output


class Layer1(nn.Module):
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
    
    def forward(self, x):
        return self.network(x)


class LayerG1(nn.Module):
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.gcn = GraphConvolution(n_in, n_out)
        
        self.network = nn.Sequential(
            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
    
    def forward(self, x, adj):
        return self.network(self.gcn(x, adj))


class Layer2(nn.Module):
    
    def __init__(
        self,
        n_in: int = 128,
        n_out: int = 10,
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.var_eps = var_eps
        self.mean_encoder = nn.Linear(n_in, n_out)
        self.var_encoder = nn.Linear(n_in, n_out)
        self.var_activation = torch.exp if var_activation is None else var_activation
        
    def forward(self, x):
        q_m = self.mean_encoder(x)
        q_v = self.var_activation(self.var_encoder(x)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = dist.rsample()
        
        return dist, latent


class LayerG2(nn.Module):
    
    def __init__(
        self,
        n_in: int = 128,
        n_out: int = 10,
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.var_eps = var_eps
        self.mean_encoder = GraphConvolution(n_in, n_out)
        self.var_encoder = nn.Linear(n_in, n_out)
        self.var_activation = torch.exp if var_activation is None else var_activation
        
    def forward(self, x, adj):
        q_m = self.mean_encoder(x, adj)
        q_v = self.var_activation(self.var_encoder(x)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = dist.rsample()
        
        return dist, latent


class Layer3(nn.Module):
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
    
    def forward(self, x):
        return self.network(x)


class Layer4(nn.Module):
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.network(x)


class VAE(nn.Module):
    
    def __init__(
        self,
        n_input: int,
        n_input_I: int,
        n_covar: int,
        n_hidden: int,
        n_latent: int,
        dropout_rate: float = 0.1,
        var_activation: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.n_covar = n_covar
        
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        
        self.layer1 = Layer1(
            n_in=n_input,
            n_out=n_hidden,
            dropout_rate=dropout_rate,
        )
        
        self.layerG1 = LayerG1(
            n_in=n_input,
            n_out=n_hidden,
            dropout_rate=dropout_rate,
        )
        
        self.layerI1 = Layer1(
            n_in=n_input_I,
            n_out=n_hidden,
            dropout_rate=dropout_rate,
        )
        
        self.layer2 = Layer2(
            n_in=n_hidden,
            n_out=n_latent,
            var_activation=var_activation,
        )
        
        self.layerG2 = LayerG2(
            n_in=n_hidden,
            n_out=n_latent,
            var_activation=var_activation,
        )
        
        self.layerI2 = Layer2(
            n_in=n_hidden,
            n_out=n_latent,
            var_activation=var_activation,
        )
        
        self.layer3 = Layer3(
            n_in=n_latent+n_covar,
            n_out=n_hidden,
            dropout_rate=dropout_rate,
        )
        
        self.layer4 = Layer4(
            n_in=n_hidden,
            n_out=n_input,
        )
        
    def inference(self, x):
        
        x1 = self.layer1(x)
        qz, z = self.layer2(x1)
        
        return dict(x1=x1, z=z, qz=qz)
    
    def inferenceG(self, x, adj):
        
        x1 = self.layerG1(x, adj)
        qz, z = self.layerG2(x1, adj)
        
        return dict(x1=x1, z=z, qz=qz)
    
    def inferenceI(self, x):
        
        x1 = self.layerI1(x)
        qz, z = self.layerI2(x1)
        
        return dict(x1=x1, z=z, qz=qz)
    
    def generative(self, z, covar):
        
        if covar is None:
            x3 = self.layer3(z)
        else:
            x3 = self.layer3(torch.cat((z, covar), dim=-1))
        
        x4 = self.layer4(x3)
        
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        
        return dict(x3=x3, x4=x4, pz=pz)
    
    def loss(
        self,
        x,
        adj,
        inference_outputs,
        inference_outputsG,
        inference_outputsI,
        generative_outputs
    ):
        
        kl_divergence_z = kl(inference_outputs['qz'], generative_outputs['pz']).sum(dim=1)
        kl_divergence_zG = kl(inference_outputsG['qz'], generative_outputs['pz']).sum(dim=1)
        kl_divergence_zI = kl(inference_outputsI['qz'], generative_outputs['pz']).sum(dim=1)
        reconst_loss = torch.norm(x - generative_outputs['x4'])
        kl_loss = torch.mean(kl_divergence_z) + torch.mean(kl_divergence_zG) + torch.mean(kl_divergence_zI)
        contrast_loss = torch.norm(inference_outputs['x1'] - inference_outputsG['x1'])
        contrast_loss += torch.norm(inference_outputs['qz'].loc - inference_outputsG['qz'].loc)
        contrast_loss += torch.norm(inference_outputs['x1'] - inference_outputsI['x1'])
        contrast_loss += torch.norm(inference_outputs['qz'].loc - inference_outputsI['qz'].loc)
        loss = reconst_loss + kl_loss / 3 + contrast_loss / 4
        
        return loss



















