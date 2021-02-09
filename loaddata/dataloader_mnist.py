import torch
import torchvision.datasets as datasets
from sklearn.metrics import pairwise_distances
import numpy as np
from loaddata.sigma import PoolRunner
import scipy
from sklearn.preprocessing import StandardScaler
from pynndescent import NNDescent

class MNIST(torch.utils.data.Dataset):
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

        self.trains=trans
        self.perplexity = perplexity
        self.v_input = v_input
        self.train = train
        self.n_point = n_point
        self.random_state = random_state
        self.root = root
        self.device = device

    def _LoadData(self):

        print('load mnist dataset')
        dataloader = datasets.MNIST(
            root="./data", train=self.train, download=True, transform=None
        )
        self.data = dataloader.data[:self.n_point].float() / 255
        self.label = dataloader.targets[:self.n_point]


    def _Pretreatment(self, ):

        if self.data.shape[0] > 5000:
            rho, sigma = self._initKNN(
                self.data, perplexity=self.perplexity, v_input=self.v_input)
        else:
            rho, sigma = self._initPairwise(
                self.data, perplexity=self.perplexity, v_input=self.v_input)
            
        self.sigma = sigma
        self.rho = rho
        self.inputdim = self.data[0].shape

    def _initPairwise(self, X, perplexity, v_input):
        
        dist = np.power(
            pairwise_distances(
                X.reshape((X.shape[0],-1)),
                n_jobs=-1),
                2)
        rho = self._CalRho(dist)

        r = PoolRunner(
            number_point = X.shape[0],
            perplexity=perplexity,
            dist=dist,
            rho=rho,
            gamma=self._CalGamma(v_input),
            v=v_input,
            pow=2)
        sigma = np.array(r.Getout())

        std_dis = np.std(rho) / np.sqrt(X.shape[1])
        print('std_dis', std_dis)
        if std_dis < 0.20:
            sigma[:] = sigma.mean() * 5
            rho[:] = 0
        
        return rho, sigma

    def _initKNN(self, X, perplexity, v_input, K=500):
        
        X_rshaped = X.reshape((X.shape[0],-1))
        index = NNDescent(X_rshaped, n_jobs=-1)
        neighbors_index, neighbors_dist = index.query(X_rshaped, k=K )
        neighbors_dist = np.power(neighbors_dist, 2)
        rho = neighbors_dist[:, 1]

        r = PoolRunner(
            number_point = X.shape[0],
            perplexity=perplexity,
            dist=neighbors_dist,
            rho=rho,
            gamma=self._CalGamma(v_input),
            v=v_input,
            pow=2)
        sigma = np.array(r.Getout())

        std_dis = np.std(rho) / np.sqrt(X.shape[1])
        print('std_dis', std_dis)
        if std_dis < 0.20:
            sigma[:] = sigma.mean() * 5
        
        return rho, sigma


    def _CalRho(self, dist):
        dist_copy = np.copy(dist)
        row, col = np.diag_indices_from(dist_copy)
        dist_copy[row,col] = 1e16
        rho = np.min(dist_copy, axis=1)
        return rho
    
    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b
        return out
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        # if index in self.data:
        data_item = self.data[index]
        rho_item = self.rho[index]
        sigma_item = self.sigma[index]
        label_item = self.label[index]

        # if self.trains is not None:
        #     data_item = self.trains(data_item)

        return index, data_item, rho_item, sigma_item, label_item