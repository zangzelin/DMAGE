from loaddata import dataloader_mnist
import torch
from sklearn.datasets import make_swiss_roll
from sklearn.metrics import pairwise_distances
import numpy as np
from loaddata.sigma import PoolRunner
import scipy
from sklearn.preprocessing import StandardScaler
from pynndescent import NNDescent

class SWISSROLL(dataloader_mnist.MNIST):
    def __init__(
        self,
        n_point,
        random_state=1,
        root='data/',
        train=True,
        trans=None,
        perplexity=None,
        v_input=100,
        device=None
    ):
        super(SWISSROLL, self).__init__(
            n_point=n_point,
            random_state=random_state,
            root=root,
            train=train,
            trans=trans,
            perplexity=perplexity,
            v_input=v_input,
            device=device,
        )


    def _LoadData(self):
        print('load SWISSROLL dataset')
        if not self.train:
            random_state = self.random_state + 1

        data = make_swiss_roll(
            n_samples=self.n_point,
            noise=0.0, 
            random_state=self.random_state
        )
        
        self.data = StandardScaler().fit_transform(data[0])
        self.label = data[1]
