import torch
import torchvision.datasets as datasets
from sklearn.metrics import pairwise_distances
import numpy as np
from loaddata.sigma import PoolRunner
import scipy
from pynndescent import NNDescent
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
import sys
import sklearn

class Cora(torch.utils.data.Dataset):
    def __init__(
        self,
        n_point,
        random_state=1,
        root='data/',
        train=True,
        trans=None,
        perplexity=None,
        v_input=100,
        device=None,
        enlarge=10
    ):

        self.trains=trans
        self.perplexity = perplexity
        self.v_input = v_input
        self.train = train
        self.n_point = n_point
        self.random_state = random_state
        self.root = root
        self.device = device
        self.data_name = 'cora'
        self.enlarge = enlarge

    def _LoadData(self):

        path_data = "./data/graphdata"
        names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
        objects = []
        for i in range(len(names)):
            with open(
                path_data + "/ind.{}.{}".format(self.data_name.lower(), names[i]), "rb"
            ) as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding="latin1"))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = self._ParseIndexFile(
            path_data + "/ind.{}.test.index".format(self.data_name))
        test_idx_range = np.sort(test_idx_reorder)

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        features = self._FeatureNormalize(features)
        data_train = torch.FloatTensor(np.array(features.todense())).float()
        labels = torch.LongTensor(labels)
        label_train = torch.max(labels, dim=1)[1]
        
        self.data = data_train
        self.label = label_train
        self.graph = graph

    def _SparseMxToTorchSparseTensor(self, sparse_mx, data_name):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        if data_name != 'wiki':
            values = torch.from_numpy(sparse_mx.data)
        else:
            values = torch.ones_like(torch.from_numpy(sparse_mx.data))
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def _FeatureNormalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        rowsum = (rowsum == 0) * 1 + rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def _ParseIndexFile(self, filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index


    def _Pretreatment(self, ):

        # if self.data.shape[0] > 5000:
        #     rho, sigma = self._initKNN(
        #         self.data, perplexity=self.perplexity, v_input=self.v_input)
        # else:
        rhoG, rhoE, PG, PE, adj = self._initPairwise(
            self.data, perplexity=self.perplexity, v_input=self.v_input)
            
        self.rhoG, self.rhoE, self.PG, self.PE =rhoG, rhoE, PG, PE
        self.adj = adj.to(self.device)
        self.inputdim = self.data[0].shape

    def EdgeSample(self, droprate):
        # if droprate >= 0:
        percent = 1-droprate
        nnz = self.adj._nnz()
        
        perm = torch.randperm(nnz)
        preserve_nnz = int(nnz*percent)
        perm = perm[:preserve_nnz]

        adjn = torch.sparse.FloatTensor(
            self.adj._indices()[:, perm],
            self.adj._values()[perm],
            self.adj.shape
        )

        adjn = self.sys_normalized_adjacency(adjn, self.device)
        return adjn

    def sys_normalized_adjacency(self, adj, tdevice):
        adj = torch.add(torch.eye(adj.shape[0], device=tdevice), adj) #adj + torch.eye(adj.shape[0])
        row_sum = torch.sum(adj, dim=1)
        row_sum=(row_sum==0)*1+row_sum
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt.mm(adj).mm(d_mat_inv_sqrt)



    def _initPairwise(self, X, perplexity, v_input):
        
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(self.graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = sp.coo_matrix(adj).tocoo()

        DisE = sklearn.metrics.pairwise.cosine_distances(self.data, self.data)
        G = nx.Graph()
        for i in range(len(self.graph)):
            G.add_node(i)
        for i in range(len(self.graph)):
            for j in self.graph[i]:
                G.add_weighted_edges_from([(i, j, DisE[i, j])])
        s = nx.to_scipy_sparse_matrix(G)
        DisG = scipy.sparse.csgraph.dijkstra(s)
        adj = self._SparseMxToTorchSparseTensor(adj, self.data_name)

        # ###############################
        copyDisG = np.copy(DisG)
        infcopyDisG = np.isposinf(copyDisG)
        copyDisG[infcopyDisG] = -100
        maxDisG = copyDisG.max(axis=0)
        for i in range(DisG.shape[0]):
            DisG[i][infcopyDisG[i]] = maxDisG[i]*self.enlarge
        

        rhoG = self._CalRho(DisG)
        sigmaG = np.array(
                PoolRunner(
                    number_point = DisG.shape[0],
                    perplexity=perplexity,
                    dist=DisG,
                    rho=rhoG,
                    gamma=self._CalGamma(v_input),
                    v=v_input,
                    pow=2
                ).Getout()
            )
        PG = self._Similarity(
            dist = DisG,
            rho = rhoG.reshape((DisG.shape[0], 1)),
            sigma_array = sigmaG.reshape((DisG.shape[0], 1)),
            gamma=self._CalGamma(v_input)
            )

        rhoE = self._CalRho(DisE)
        sigmaE = np.array(
                PoolRunner(
                    number_point = DisE.shape[0],
                    perplexity=perplexity,
                    dist=DisE,
                    rho=rhoE,
                    gamma=self._CalGamma(v_input),
                    v=v_input,
                    pow=2
                ).Getout()
            )
        PE = self._Similarity(
            dist = DisE,
            rho = rhoE.reshape((DisE.shape[0], 1)),
            sigma_array = sigmaE.reshape((DisE.shape[0], 1)),
            gamma=self._CalGamma(v_input)
            )


        return rhoG, rhoE, PG, PE, adj

    def _Similarity(self, dist, rho, sigma_array, gamma, v=100):

        # if torch.is_tensor(rho):
        dist_rho = (dist - rho) / sigma_array
        dist_rho[dist_rho < 0] = 0

        Pij = np.power(
            gamma * np.power(
                (1 + dist_rho / v),
                -1 * (v + 1) / 2
                ) * np.sqrt(2 * 3.14),
                2
            )

        P = Pij + Pij.T - np.multiply(Pij, Pij.T)

        return P

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
        PG_item = self.PG[index]
        PE_item = self.PE[index]
        label_item = self.label[index]

        # if self.trains is not None:
        #     data_item = self.trains(data_item)

        return index, data_item,  PG_item, PE_item, label_item