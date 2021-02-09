# a pytorch based lisv2 code
# author: zelinzang
# email: zangzelin@gmail.com

import functools
import pdb
import time
from locale import currency
from multiprocessing import Pool
from typing import Any

import numpy as np
import torch
import torch.autograd
from torch import nn, set_flush_denormal
# import  as RestNet
from model.FCA import FCA


class LISV2_Model(torch.nn.Module):
    def __init__(
        self,
        input_dim: list,
        device: Any,
        NetworkStructure: list,
        dataset_name: str,
        model_type: str,
        latent_dim=2,
        num_point=1000,
        input_data=None,
    ):

        super(LISV2_Model, self).__init__()
        with torch.no_grad():
            self.device = device
            self.phis = []
            self.n_dim = latent_dim
            self.NetworkStructure = NetworkStructure
            self.NetworkStructure[0] = functools.reduce(lambda x,y:x * y, input_dim)
            self.dataset_name = dataset_name
            self.model_type = model_type

            self.InitNetworkMLP()
 
        
        print(self.encoder)


    def InitNetworkMLP(self):

        self.encoder = nn.ModuleList()
        for i in range(len(self.NetworkStructure) - 1):
            self.encoder.append(
                nn.Linear(self.NetworkStructure[i],
                          self.NetworkStructure[i + 1]))
            if i != len(self.NetworkStructure) - 2:
                self.encoder.append(nn.LeakyReLU(0.1))
        self.FCA = FCA(250, 250)

    def GetInput(self):
        return None

    def forward(self, x, adj=None, index=None):

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i == 3:
                x = self.FCA(x, adj)

        return x

    def Generate(self, latent, index=None):

        x = latent.reshape(latent.shape[0], -1)
        for i, layer in enumerate(self.decoder):
            x = layer(x)

        return [x]

    def test(self, input_data):

        self.loss.ChangeVList()
        x = input_data.to(self.device)

        for i, layer in enumerate(self.encoder):
            x = layer(x)

        return [x]