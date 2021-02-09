# from baseline import Distance_squared
import math
import multiprocessing
import os
from re import L
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC

# from plotfig_plotpy import FindContainList

warnings.filterwarnings("ignore")

def NNACC(calc, label):
    indexNN = calc.neighbours_Z[:, 0].reshape(-1)
    labelNN = label[indexNN]
    acc = (label == labelNN).sum() / label.shape[0]
    # print(acc)
    # input()
    return acc

def DistanceAUC(disZ, label):
    from sklearn import metrics

    disZN = (disZ-disZ.min())/(disZ.max()-disZ.min())
    LRepeat = label.reshape(1,-1).repeat(disZ.shape[0], axis=0)
    L = (LRepeat==LRepeat.T).reshape(-1)
    auc = metrics.roc_auc_score(1-L, disZN.reshape(-1))
    
    return auc

def GetIndicatorData(
    real_data, latent, label=None, KNN=5, kmax=10, save_path=None, Name="", dist=None
):

    calc = MeasureCalculator(real_data, latent, kmax + 1, dist=dist)

    # rmse_local = []
    mrreZX = []
    mrreXZ = []
    cont = []
    trust = []

    for k in range(int(kmax * 0.6), kmax, int(kmax * 0.1)):
        mrreZX.append(calc.mrre(k)[0])
        mrreXZ.append(calc.mrre(k)[1])
        cont.append(calc.continuity(k))
        trust.append(calc.trustworthiness(k))

    indicator = {}
    indicator["name"] = Name
    # indicator["mrre ZX"] = float(np.mean(mrreZX))
    # indicator["mrre XZ"] = float(np.mean(mrreXZ))
    indicator["mrre"] = float((np.mean(mrreXZ) + np.mean(mrreZX)) / 2)
    indicator["cont"] = float(np.mean(cont))
    indicator["trust"] = float(np.mean(trust))
    indicator['kl'] = calc.density_kl_global(sigma=2.43)
    print(indicator['kl'])

    if label is not None:
        acc, std = TestClassifacation(latent, label)
        indicator["ACC"] = float(acc)
        indicator["std"] = float(std)
        indicator["NNACC"] = NNACC(calc=calc, label=label)
        indicator["AUC"] = float(DistanceAUC(calc.pairwise_Z, label=label))

    else:
        indicator["ACC"] = 0
        indicator["std"] = 0
        indicator["NNACC"] = 0
        indicator["AUC"] = 0

    # print(indicator)
    if save_path is not None:
        if not os.path.exists(save_path):
            print(", ".join(list(indicator.keys())), file=open(save_path, "a"))
        print(
            ", ".join(map(lambda x: str(x), list(indicator.values()))),
            file=open(save_path, "a"),
        )
        # print('ppp')
    else:
        print(
            ", ".join(map(lambda x: str(x), list(indicator.values()))),
            file=open(save_path, "a"),
        )

    return indicator


class MeasureRegistrator:
    """Keeps track of measurements in Measure Calculator."""

    k_independent_measures = {}
    k_dependent_measures = {}

    def register(self, is_k_dependent):
        def k_dep_fn(measure):
            self.k_dependent_measures[measure.__name__] = measure
            return measure

        def k_indep_fn(measure):
            self.k_independent_measures[measure.__name__] = measure
            return measure

        if is_k_dependent:
            return k_dep_fn
        return k_indep_fn

    def get_k_independent_measures(self):
        return self.k_independent_measures

    def get_k_dependent_measures(self):
        return self.k_dependent_measures


def Distance_squared(x, y):
    import torch

    x = torch.tensor(x).cuda(2)
    y = torch.tensor(y).cuda(2)
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    d = dist.clamp(min=1e-36)
    return d.detach().cpu().numpy()


class MeasureCalculator:
    measures = MeasureRegistrator()

    def __init__(self, Xi, Zi, k_max, dist=None):
        self.k_max = k_max
        if torch.is_tensor(Xi):
            self.X = Xi.detach().cpu().numpy()
            self.Z = Zi.detach().cpu().numpy()
        else:
            self.X = Xi
            self.Z = Zi
        self.pairwise_Z = np.sqrt(Distance_squared(Zi, Zi))
        self.pairwise_Z = (self.pairwise_Z - self.pairwise_Z.min()) / (
            self.pairwise_Z.max() - self.pairwise_Z.min()
        )
        if dist is not None:
            self.pairwise_X = np.sqrt(dist)
        else:
            self.pairwise_X = squareform(pdist(self.X))
        self.pairwise_X = (self.pairwise_X - self.pairwise_X.min()) / (
            self.pairwise_X.max() - self.pairwise_X.min()
        )
        # self.pairwise_Z = self.pairwise_Z.max()

        self.neighbours_X, self.ranks_X = self._neighbours_and_ranks(
            self.pairwise_X, k_max
        )
        self.neighbours_Z, self.ranks_Z = self._neighbours_and_ranks(
            self.pairwise_Z, k_max
        )
        # print(self.ranks_X.shape)
        # print(self.neighbours_X.shape)
        # input()

    #         print('finish init')

    @staticmethod
    def _neighbours_and_ranks(distances, k):
        """
        Inputs:
        - distances,        distance matrix [n times n],
        - k,                number of nearest neighbours to consider
        Returns:
        - neighbourhood,    contains the sample indices (from 0 to n-1) of kth nearest neighbor of current sample [n times k]
        - ranks,            contains the rank of each sample to each sample [n times n], whereas entry (i,j) gives the rank that sample j has to i (the how many 'closest' neighbour j is to i)
        """
        # Warning: this is only the ordering of neighbours that we need to
        # extract neighbourhoods below. The ranking comes later!
        indices = np.argsort(distances, axis=-1, kind="stable")

        # Extract neighbourhoods.
        neighbourhood = indices[:, 1 : k + 1]

        # Convert this into ranks (finally)
        ranks = indices.argsort(axis=-1, kind="stable")
        # print(ranks)

        return neighbourhood, ranks

    def get_X_neighbours_and_ranks(self, k):
        return self.neighbours_X[:, :k], self.ranks_X

    def get_Z_neighbours_and_ranks(self, k):
        return self.neighbours_Z[:, :k], self.ranks_Z

    def compute_k_independent_measures(self):
        return {
            key: fn(self)
            for key, fn in self.measures.get_k_independent_measures().items()
        }

    def compute_k_dependent_measures(self, k):
        return {
            key: fn(self, k)
            for key, fn in self.measures.get_k_dependent_measures().items()
        }

    def compute_measures_for_ks(self, ks):
        return {
            key: np.array([fn(self, k) for k in ks])
            for key, fn in self.measures.get_k_dependent_measures().items()
        }

    @measures.register(False)
    def stress(self):
        sum_of_squared_differences = np.square(self.pairwise_X - self.pairwise_Z).sum()
        sum_of_squares = np.square(self.pairwise_Z).sum()

        return np.sqrt(sum_of_squared_differences / sum_of_squares)

    @measures.register(False)
    def rmse(self):
        n = self.pairwise_X.shape[0]
        sum_of_squared_differences = np.square(self.pairwise_X - self.pairwise_Z).sum()
        return np.sqrt(sum_of_squared_differences / n ** 2)

    @measures.register(False)
    def local_rmse(self, k):
        X_neighbors, _ = self.get_X_neighbours_and_ranks(k)
        mses = []
        n = self.pairwise_X.shape[0]
        for i in range(n):
            x = self.X[X_neighbors[i]]
            z = self.Z[X_neighbors[i]]
            d1 = self.neighbours_X
            d2 = self.neighbours_Z

            d1 = (d1 - d1.min()) / (d1.max() - d1.min())
            d2 = (d2 - d2.min()) / (d2.max() - d2.min())

            mse = np.sum(np.square(d1 - d2))
            mses.append(mse)
        return np.sqrt(np.sum(mses) / (k * n))

    @staticmethod
    def _trustworthiness(X_neighbourhood, X_ranks, Z_neighbourhood, Z_ranks, n, k):
        """
        Calculates the trustworthiness measure between the data space `X`
        and the latent space `Z`, given a neighbourhood parameter `k` for
        defining the extent of neighbourhoods.
        """

        result = 0.0

        # Calculate number of neighbours that are in the $k$-neighbourhood
        # of the latent space but not in the $k$-neighbourhood of the data
        # space.
        for row in range(X_ranks.shape[0]):
            missing_neighbours = np.setdiff1d(
                Z_neighbourhood[row], X_neighbourhood[row]
            )

            for neighbour in missing_neighbours:
                result += X_ranks[row, neighbour] - k

        return 1 - 2 / (n * k * (2 * n - 3 * k - 1)) * result

    @measures.register(True)
    def trustworthiness(self, k):
        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)
        n = self.pairwise_X.shape[0]
        return self._trustworthiness(
            X_neighbourhood, X_ranks, Z_neighbourhood, Z_ranks, n, k
        )

    @measures.register(True)
    def continuity(self, k):
        """
        Calculates the continuity measure between the data space `X` and the
        latent space `Z`, given a neighbourhood parameter `k` for setting up
        the extent of neighbourhoods.

        This is just the 'flipped' variant of the 'trustworthiness' measure.
        """

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)
        n = self.pairwise_X.shape[0]
        # Notice that the parameters have to be flipped here.
        return self._trustworthiness(
            Z_neighbourhood, Z_ranks, X_neighbourhood, X_ranks, n, k
        )

    @measures.register(True)
    def neighbourhood_loss(self, k):
        """
        Calculates the neighbourhood loss quality measure between the data
        space `X` and the latent space `Z` for some neighbourhood size $k$
        that has to be pre-defined.
        """

        X_neighbourhood, _ = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, _ = self.get_Z_neighbours_and_ranks(k)

        result = 0.0
        n = self.pairwise_X.shape[0]

        for row in range(n):
            shared_neighbours = np.intersect1d(
                X_neighbourhood[row], Z_neighbourhood[row], assume_unique=True
            )

            result += len(shared_neighbours) / k

        return 1.0 - result / n

    @measures.register(True)
    def rank_correlation(self, k):
        """
        Calculates the spearman rank correlation of the data
        space `X` with respect to the latent space `Z`, subject to its $k$
        nearest neighbours.
        """

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)

        n = self.pairwise_X.shape[0]
        # we gather
        gathered_ranks_x = []
        gathered_ranks_z = []
        for row in range(n):
            # we go from X to Z here:
            for neighbour in X_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]
                gathered_ranks_x.append(rx)
                gathered_ranks_z.append(rz)
        rs_x = np.array(gathered_ranks_x)
        rs_z = np.array(gathered_ranks_z)
        coeff, _ = spearmanr(rs_x, rs_z)

        # use only off-diagonal (non-trivial) ranks:
        # inds = ~np.eye(X_ranks.shape[0],dtype=bool)
        # coeff, pval = spearmanr(X_ranks[inds], Z_ranks[inds])
        return coeff

    @measures.register(True)
    def mrre(self, k):
        """
        Calculates the mean relative rank error quality metric of the data
        space `X` with respect to the latent space `Z`, subject to its $k$
        nearest neighbours.
        """

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)

        n = self.pairwise_X.shape[0]

        # First component goes from the latent space to the data space, i.e.
        # the relative quality of neighbours in `Z`.

        mrre_ZX = 0.0
        for row in range(n):
            for neighbour in Z_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]

                mrre_ZX += abs(rx - rz) / rz

        # Second component goes from the data space to the latent space,
        # i.e. the relative quality of neighbours in `X`.

        mrre_XZ = 0.0
        for row in range(n):
            # Note that this uses a different neighbourhood definition!
            for neighbour in X_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]

                # Note that this uses a different normalisation factor
                mrre_XZ += abs(rx - rz) / rx

        # Normalisation constant
        C = n * sum([abs(2 * j - n - 1) / j for j in range(1, k + 1)])
        return mrre_ZX / C, mrre_XZ / C

    @measures.register(False)
    def density_global(self, sigma=0.1):
        X = self.pairwise_X
        X = X / X.max()
        Z = self.pairwise_Z
        Z = Z / Z.max()

        density_x = np.sum(np.exp(-(X ** 2) / sigma), axis=-1)
        density_x /= density_x.sum(axis=-1)

        density_z = np.sum(np.exp(-(Z ** 2) / sigma), axis=-1)
        density_z /= density_z.sum(axis=-1)

        return np.abs(density_x - density_z).sum()

    @measures.register(False)
    def density_kl_global(self, sigma=0.1):
        X = self.pairwise_X
        X = X / X.max()
        Z = self.pairwise_Z
        Z = Z / Z.max()

        density_x = np.sum(np.exp(-(X ** 2) / sigma), axis=-1)
        density_x /= density_x.sum(axis=-1)

        density_z = np.sum(np.exp(-(Z ** 2) / sigma), axis=-1)
        density_z /= density_z.sum(axis=-1)

        return (density_x * (np.log(density_x) - np.log(density_z))).sum()

    @measures.register(False)
    def density_kl_global_10(self):
        return self.density_kl_global(10.0)

    @measures.register(False)
    def density_kl_global_1(self):
        return self.density_kl_global(1.0)

    @measures.register(False)
    def density_kl_global_01(self):
        return self.density_kl_global(0.1)

    @measures.register(False)
    def density_kl_global_001(self):
        return self.density_kl_global(0.01)

    @measures.register(False)
    def density_kl_global_0001(self):
        return self.density_kl_global(0.001)


def TestClassifacation(embedding, label):

    method = SVC(kernel="linear", max_iter=90000)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    # if
    n_scores = cross_val_score(
        method, embedding, label, scoring="accuracy", cv=cv, n_jobs=-1
    )

    return (
        n_scores.mean(),
        n_scores.std(),
    )


def TestOtherMehtod(dataname, path, name, savePath):

    dirlist = os.listdir(path)
    dirlist.sort()

    dirlistI = FindContainList(FindContainList(dirlist, dataname), name)
    pathin = FindContainList(FindContainList(dirlistI, "data_train"), name)
    pathem = FindContainList(FindContainList(dirlistI, "n2em"), name)
    pathla = FindContainList(FindContainList(dirlistI, "label_train"), name)
    pathdis = FindContainList(FindContainList(dirlistI, "dis"), name)

    data_in = np.load(path + "/" + pathin[0])
    data_em = np.load(path + "/" + pathem[0])
    data_la = np.load(path + "/" + pathla[0])
    data_dis = np.load(path + "/" + pathdis[0])

    if data_in.shape[0] > 10000:
        subsample_index = np.random.choice(range(data_in.shape[0]), 6000, replace=False)
        data_in = data_in[subsample_index]
        data_em = data_em[subsample_index]
        data_la = data_la[subsample_index]
        data_dis = data_dis[subsample_index, :][:, subsample_index]
    if dataname == "Spheres5500" or dataname == "Spheres10000":
        kmax = data_in.shape[0] // 7
    # elif dataname == 'coil20' or dataname == 'coil100':
    #     kmax = int(data_in.shape[0] *0.75)
    else:
        kmax = 10
    name = name + "_" + dataname

    if (
                dataname == "swishroll" or dataname == "severedsphere" or dataname == "SCurve" 
    ):
        indi = GetIndicatorData(
            real_data=data_in,
            latent=data_em,
            # label=data_la,
            save_path=savePath + "lisind.csv",
            Name=name,
            dist=data_dis,
            kmax=kmax,
        )
    else:
        indi = GetIndicatorData(
            real_data=data_in,
            latent=data_em,
            label=data_la,
            save_path=savePath + "lisind.csv",
            Name=name,
            dist=data_dis,
            kmax=kmax,
        )


def TestElisMehtod(dataname, savePath, path="log"):

    dirlist = os.listdir(path)
    dirlist.sort()
    # dirlist = ["20200915082316_68fd5sphere5500_T"]

    dirlist = FindContainList(dirlist, dataname)

    for dirItem in dirlist:
        p = path + "/" + dirItem
        print(p)

        listname = os.listdir(p)
        listContainInput = FindContainList(listname, "input.npy")
        listContainLabel = FindContainList(listname, "label.npy")
        listContainlLatent = FindContainList(listname, "latent.npy")
        listContainDist = FindContainList(listname, "dist.npy")

        listContainlLatent.sort()

        for i, latent_p in enumerate(listContainlLatent[-10:]):
            print("start to add {}".format(p + "/" + latent_p))
            # print()
            input_data_path = p + "/" + listContainInput[0]
            latent_path = p + "/" + latent_p
            label_path = p + "/" + listContainLabel[0]
            # dist_path = p + "/" + listContainDist[0]

            data_in = np.load(input_data_path)
            data_em = np.load(latent_path)
            data_la = np.load(label_path)
            # data_dis = np.load(dist_path)

            if data_in.shape[0] > 10000:
                subsample_index = np.random.choice(
                    range(data_in.shape[0]), 6000, replace=False
                )
                data_in = data_in[subsample_index]
                data_em = data_em[subsample_index]
                data_la = data_la[subsample_index]
                # data_dis = data_dis[subsample_index, :][:, subsample_index]

            if dataname == "Spheres5500" or dataname == "Spheres10000":
                kmax = data_in.shape[0] // 7
            # elif dataname == 'coil20' or dataname == 'coil100':
            #     kmax = int(data_in.shape[0] *0.75)
            else:
                kmax = 10

            name = "Elis_" + dataname + latent_p
            print("finish loading")
            if (
                dataname == "swishroll" or dataname == "severedsphere" or dataname == "SCurve"
            ):
                indi = GetIndicatorData(
                    real_data=data_in,
                    latent=data_em,
                    # label=data_la,
                    save_path=savePath + "lisind.csv",
                    Name=name,
                    # dist=data_dis,
                    kmax=kmax,
                )
            else:
                indi = GetIndicatorData(
                    real_data=data_in,
                    latent=data_em,
                    label=data_la,
                    save_path=savePath + "lisind.csv",
                    Name=name,
                    # dist=data_dis,
                    kmax=kmax,
                )


def main():
    import numpy as np

    savePath = "result/v2/"
    path = "result/other"
    # methodname = "tsne"
    datanameList = [
        # "swishroll",
        # "severedsphere",
        # "SCurve",
        # "swishroll",
        # "Spheres5500",
        "Spheres10000",
        # # "coil20",
        # # "coil100",
        # "emnist",
        # "Fmnist",
        # "cifa",

    ]

    for dataname in datanameList:
        print(dataname)
        # TestElisMehtod(dataname=dataname,savePath='result/v2/')
        TestOtherMehtod(dataname=dataname, path=path, name="MLLE", savePath=savePath)
        # TestOtherMehtod(dataname=dataname, path=path, name="tsne", savePath=savePath)
        # TestOtherMehtod(dataname=dataname, path=path, name="umap", savePath=savePath)



if __name__ == "__main__":

    main()
