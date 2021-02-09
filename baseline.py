import os
import sys

import numpy as np
import sklearn
import torch
from PIL import Image
from sklearn.datasets import fetch_openml, make_s_curve, make_swiss_roll

import indicator
import tool
from dataloader import GetData
import main
import tool

sys.path.append("../")


def Test(clf, data_train, args):

    em = clf.fit_transform(data_train)
    print("finish fit")

    return em


def Model(modelName, args):
    if modelName == "umap":
        import umap

        clf = umap.UMAP(
            n_components=min(args["n_components"], args["dataDimention"]),
            random_state=1,
            n_neighbors=args['perplexity']
        )
    if modelName == "tsne":
        from sklearn.manifold import TSNE

        print("nnnn", min(args["n_components"], args["dataDimention"]))
        clf = TSNE(
            n_components=min(args["n_components"], args["dataDimention"]),
            # method="exact",
            random_state=1,
        )
    if modelName == "PCA":
        from sklearn.decomposition import PCA

        clf = PCA(
            n_components=min(args["n_components"], args["dataDimention"]),
            random_state=1,
        )
    if modelName == "ISOMAP":
        from sklearn.manifold import Isomap

        clf = Isomap(n_components=min(args["n_components"], args["dataDimention"]))
    if modelName == "SC":
        from sklearn.manifold import SpectralEmbedding

        clf = SpectralEmbedding(
            n_components=min(args["n_components"], args["dataDimention"]),
            random_state=1,
        )
    if modelName == "LLE":
        from sklearn.manifold import LocallyLinearEmbedding

        clf = LocallyLinearEmbedding(
            n_components=min(args["n_components"], args["dataDimention"]),
            random_state=1,
        )
    if modelName == "MLLE":
        from sklearn.manifold import LocallyLinearEmbedding

        clf = LocallyLinearEmbedding(
            n_neighbors=10,
            n_components=2,
            method="modified",
        )
    if modelName == "HLLE":
        from sklearn.manifold import LocallyLinearEmbedding

        clf = LocallyLinearEmbedding(
            n_neighbors=500,
            n_components=min(args["n_components"], args["dataDimention"]),
            method="hessian",
            random_state=1,
            max_iter=100,
        )
    if modelName == "LTSA":
        from sklearn.manifold import LocallyLinearEmbedding

        clf = LocallyLinearEmbedding(
            n_neighbors=min(args["n_components"], args["dataDimention"]) + 1,
            n_components=min(args["n_components"], args["dataDimention"]),
            method="ltsa",
            random_state=1,
            max_iter=100,
        )
    if modelName == "MDS":
        from sklearn.manifold import MDS

        clf = MDS(
            n_components=min(args["n_components"], args["dataDimention"]),
            random_state=1,
            max_iter=100,
        )

    return clf


def Distance_squared(x, y):
    import torch

    x = torch.tensor(x)#.cuda(4)
    y = torch.tensor(y)#.cuda(4)
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    d = dist.clamp(min=1e-36)
    return d.detach().cpu().numpy()


def SavetotxtLarvis(data, path="largis.txt"):

    f = open(path, "w")
    f = open(path, "a")
    f.write("{} {}\n".format(str(data.shape[0]), str(data.shape[1])))
    for i in range(data.shape[0]):
        datalist = data[i].tolist()
        datastr = [str(s) for s in datalist]
        f.write(" ".join(datastr) + "\n")
    f.close()


def TestMethod(args, path="./baseline/"):
    device = torch.device("cpu")
    data_train, label_train = GetData(args, device=device,)
    data_train = data_train.numpy()
    label_train = label_train.numpy()
    # print(data_train.shape)
    # input()
    args["dataDimention"] = data_train.shape[1]
    print(data_train.shape)

    if args["method"] is not "LargeVis":
        clf = Model(args["method"], args)
        em = Test(clf, data_train, args)
    else:
        SavetotxtLarvis(data_train)
        result = os.popen(
            "/root/LargeVis/Linux/LargeVis -input largis.txt -output largisout.txt"
        )
        context = result.read()
        em = np.loadtxt("largisout.txt", delimiter=" ", skiprows=1)

    name = "name{}_method{}_k{}_n{}".format(
        args["data_name"], args["method"], args["perplexity"], args["n_components"]
    )

    dis = Distance_squared(data_train, data_train)

    np.save(path + name + "em.npy", em)
    np.save(path + name + "data_train.npy", data_train)
    np.save(path + name + "label_train.npy", label_train)
    np.save(path + name + "dis.npy", dis)


    ploter = tool.GIFPloter()
    ploter.AddNewFig(
        em,
        label_train,
        title_=name + ".png",
        path=path,
    )

    if em.shape[0] > 8000:
        subsample_index = np.random.choice(range(em.shape[0]), 6000, replace=False)
        em = em[subsample_index]
        data_train = data_train[subsample_index]
        label_train = label_train[subsample_index]
        dis = dis[subsample_index, :][:, subsample_index]

    if label_train.dtype in [np.int, np.int32, np.int64]:
        indi = indicator.GetIndicatorData(
            real_data=data_train,
            latent=em,
            label=label_train,
            save_path=path + name + "inid.csv",
            dist=dis,
            Name=name,
        )
    else:
        indi = indicator.GetIndicatorData(
            real_data=data_train,
            latent=em,
            # label=label_train,
            save_path=path + name + "inid.csv",
            dist=dis,
            Name=name,
        )


if __name__ == "__main__":

    args = {
        "data_name": "coli20",
        "perplexity": 15,
        "method": "umap",
        "data_trai_n": 40000,
        "n_components": 2,
    }

    dataList = [
        "Duplicate"
        # "swishroll",
        # "severedsphere",
        # # "SCurve",
        # 'Spheres5500',
        # 'Spheres10000',
        # "coil20",
        # "coil100rgb",
        # "mnist",
        # "Fmnist",
        # "cifa10",
    ]

    # methodList = ["PCA", "MLLE", "tsne", "umap"]
    methodList = ["umap"]
    # methodList = ["MLLE"]

    n_componentsList = [2]

    for n in n_componentsList:
        args["n_components"] = n
        for dataset in dataList:
            args["data_name"] = dataset
            for mehtod in methodList:
                args["method"] = mehtod
                print(
                    "------------->",
                    args["n_components"],
                    args["method"],
                    args["data_name"],
                )
                TestMethod(args, path="result/other/")

