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

class COIL100(dataloader_mnist.MNIST):
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
        super(COIL100, self).__init__(
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
        print('load COIL100 dataset')

        path = "./data/coil-100"
        fig_path = os.listdir(path)

        label = []
        data = np.zeros((100 * 72, 64, 64, 3))
        for i, path_i in enumerate(fig_path):
            print(i)
            if "obj" in path_i:
                I = Image.open(path + "/" + path_i)
                I_array = np.array(I.resize((64, 64)))
                data[i] = I_array
                label.append(int(fig_path[i].split("__")[0].split("obj")[1]))

        self.data = np.swapaxes(data, 1, 3) / 255
        self.label = np.array(label)
