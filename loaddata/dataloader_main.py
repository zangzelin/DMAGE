from loaddata import dataloader_coil20
from tkinter.messagebox import NO
import loaddata.dataloader_swishroll as dataloader_swishroll
import loaddata.dataloader_scurve as dataloader_scurve
import loaddata.dataloader_digits as dataloader_digits
import loaddata.dataloader_mnist as dataloader_mnist
import loaddata.dataloader_coil20 as dataloader_coil20
import loaddata.dataloader_coil100 as dataloader_coil100
import torchvision.transforms as transforms
import torch

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
  
    if data_name == 'swishroll':
        dataset = dataloader_swishroll.SWISSROLL(
            n_point=n_point,
            random_state=random_state,
            trans=trans,
            perplexity=perplexity,
            device=device
            )
    if data_name == 'scurve':
        dataset = dataloader_scurve.SCURVE(
            n_point=n_point,
            random_state=random_state,
            trans=trans,
            perplexity=perplexity,
            device=device
            )
    if data_name == 'digits':
        dataset = dataloader_digits.DIGITS(
            n_point=n_point,
            random_state=random_state,
            trans=trans,
            perplexity=perplexity,
            device=device
            )
    if data_name == 'mnist':
        dataset = dataloader_mnist.MNIST(
            n_point=n_point,
            random_state=random_state,
            trans=trans,
            perplexity=perplexity,
            device=device
            )
    if data_name == 'coil20':
        dataset = dataloader_coil20.COIL20(
            n_point=n_point,
            random_state=random_state,
            trans=trans,
            perplexity=perplexity,
            device=device
            )
    if data_name == 'coil100':
        dataset = dataloader_coil100.COIL100(
            n_point=n_point,
            random_state=random_state,
            trans=trans,
            perplexity=perplexity,
            device=device
            )
    dataset._LoadData()
    dataset._Pretreatment()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        )

    return loader, dataset