from loaddata import dataloader_coil20
from tkinter.messagebox import NO
import loaddataGraph.dataloader_cora as dataloader_cora
import loaddataGraph.dataloader_citeseer as dataloader_citeseer
import loaddataGraph.dataloader_pubmed as dataloader_pubmed
import torchvision.transforms as transforms
import torch
import os
import joblib

def Getdataloader(
    data_name,
    n_point,
    batch_size,
    perplexity=None,
    num_workers=1,
    sampler=None,
    random_state=1,
    device=None,
    ):
    
    trans = transforms.Compose(
        [transforms.ToTensor()]
    )
  
    if data_name == 'cora':
        dataset = dataloader_cora.Cora(
            n_point=n_point,
            random_state=random_state,
            trans=trans,
            perplexity=perplexity,
            device=device
            )
    if data_name == 'citeseer':
        dataset = dataloader_citeseer.Citeseer(
            n_point=n_point,
            random_state=random_state,
            trans=trans,
            perplexity=perplexity,
            device=device
            )
    if data_name == 'pubmed':
        dataset = dataloader_pubmed.PubMed(
            n_point=n_point,
            random_state=random_state,
            trans=trans,
            perplexity=perplexity,
            device=device
            )

    loadpath = 'save/DATAn_point{}_data_name{}.pkl'.format(n_point, data_name)
    if os.path.exists(loadpath):
        dataset.data, dataset.label, dataset.graph = joblib.load(loadpath)
        print('load', loadpath)
    else:
        dataset._LoadData()
        joblib.dump([dataset.data, dataset.label, dataset.graph], loadpath)
        print('save', loadpath)

    loadpath = 'save/Similn_point{}_data_name{}_perplexity{}.pkl'.format(
        n_point, data_name, perplexity, )
    if os.path.exists(loadpath):
        dataset.PG, dataset.PE, dataset.adj = joblib.load(loadpath)
        dataset.inputdim = dataset.data[0].shape
        print('load', loadpath)
    else:
        dataset._Pretreatment()
        joblib.dump([dataset.PG, dataset.PE, dataset.adj], loadpath)
        print('save', loadpath)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        )
    

    return loader, dataset