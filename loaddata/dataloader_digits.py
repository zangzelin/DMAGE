from loaddata import dataloader_mnist
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances
import numpy as np
from loaddata.sigma import PoolRunner
import scipy
from sklearn.preprocessing import StandardScaler
from pynndescent import NNDescent

class DIGITS(dataloader_mnist.MNIST):
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
        super(DIGITS, self).__init__(
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

        digit = load_digits()
        self.data = digit.data
        self.label = digit.target
        
        # self.data = StandardScaler().fit_transform(digit[0])
        # self.label = data[1]
