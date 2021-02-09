from loaddata import dataloader_mnist
import torch
from sklearn.datasets import make_swiss_roll
from sklearn.metrics import pairwise_distances
import numpy as np
from loaddata.sigma import PoolRunner
import scipy
from sklearn.preprocessing import StandardScaler
from pynndescent import NNDescent
import os
from PIL import Image

class COIL20(dataloader_mnist.MNIST):
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
        super(COIL20, self).__init__(
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
        print('load COIL20 dataset')

        path = "./data/coil-20-proc"
        fig_path = os.listdir(path)
        fig_path.sort()

        label = []
        data = np.zeros((1440, 64, 64))
        for i in range(1440):
            I = Image.open(path + "/" + fig_path[i]).resize((64,64))
            I_array = np.array(I)
            data[i] = I_array
            label.append(int(fig_path[i].split("__")[0].split("obj")[1]))
        
        self.data = data / 255
        self.label = label
