from typing import Any
import numpy as np
import torch
from torch._C import device
import torch.optim as optim
from multiprocessing import Pool

# from Loss.dmt_loss_twowaydivergenceLoss import MyLoss
from Loss.dmt_loss_sourcegraph import Source
# from Loss.dmt_loss_L2 import MyLoss
# from Loss.dmt_loss_L3 import MyLoss
# from Loss.dmt_loss_ItakuraSaito import MyLoss
# from Loss.dmt_loss_GID import MyLoss


# import model.dmt_model as model_use
import model.dmage_model as model_use
# import model.dmt_model_unparam as model_use

# import loaddata.dataloader_main as dataloader_main
import loaddataGraph.dataloader_main as dataloader_main

# import param.param_dmt as paramzzl
import param.param_dmage as paramuse


from torch import nn, set_flush_denormal
from tqdm import trange
import nuscheduler

import tool
# import resource
# soft, hard = resource.getrlimit(resource.RLIMIT_AS)
# resource.setrlimit(resource.RLIMIT_AS, (int(0.5 * 1024 ** 6), hard))
import eval
torch.set_num_threads(1)

def train(
        Model: Any,
        Loss: Any,
        nushadular:Any,
        dataloader: Any,
        optimizer: Any,
        current_epoch: int,
        adj: Any,
        ):

    # print(current_epoch)
    Model.train()

    loss_list = []
    for batch_idx, (index_data, input_data, PG, PE, label_data) in enumerate(dataloader):
        
        input_data_model = input_data.reshape((input_data.shape[0], -1))
        index_data = index_data.long().to(Model.device)
        
        loss = Loss(
            # input_data=input_data_loss,
            latent_data=Model(
                input_data_model.float().to(Model.device), 
                adj=adj[index_data][:,index_data], 
                index=index_data),
            PG=PG[:, index_data].float().to(Model.device),
            PE=PE[:, index_data].float().to(Model.device),
            v_latent=nushadular.Getnu(current_epoch)
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().cpu().numpy())
    return np.sum(loss_list)


def Test(
        Model: Any,
        Loss: Any,
        dataloader: Any,
        optimizer: Any,
        current_epoch: int,
        adj: Any,
        ):

    Model.eval()
    for batch_idx, (index_data, input_data, PG, PE, label_data) in enumerate(dataloader):

        input_data_model = input_data.reshape((input_data.shape[0], -1))

        index_data = index_data.long().to(Model.device)
        input_data_model = input_data_model.float().to(Model.device)
        PG = PG[:, index_data].float().to(Model.device)
        PE = PE[:, index_data].float().to(Model.device)
        adj_b = adj[index_data][:,index_data]

        latent_data = Model(input_data_model, adj=adj_b, index=index_data)

        em = latent_data.detach().cpu().numpy()
        la = label_data.detach().cpu().numpy()
        # re = re[-1].detach().cpu().numpy()
        if batch_idx == 0:
            outem = em
            outla = la
        else:
            outem = np.concatenate((outem, em), axis=0)
            outla = np.concatenate((outla, la), axis=0)


    return outem, outla


def main(args: dict, data_train=None, label_train=None, data_test=None, label_test=None):

    tool.SetSeed(args['seed'])
    path = tool.GetPath(args['data_name']+'_'+args['name'])
    tool.SaveParam(path, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader, dataset = dataloader_main.Getdataloader(
        data_name=args['data_name'],
        n_point=args['data_trai_n'],
        batch_size=args['batch_size'],
        perplexity=args['perplexity'],
        device=device,
        )
    nushadular = nuscheduler.Nushadular(
        nu_start=args['vtrace'][0],
        nu_end=args['vtrace'][1],
        epoch_start=args['epochs']//10,
        epoch_end=args['epochs']*4//5,
    )
    
    # # Loss = dmt_loss_fast.LISV2_Loss(data_train, device=device, args=args, path=path).to(device)
    Loss = Source(
        v_input=100, device=device, alpha=args['alpha']).to(device)
    Model = model_use.LISV2_Model(
        input_dim=dataset.inputdim,
        device=device,
        NetworkStructure=args['NetworkStructure'],
        dataset_name=args['data_name'],
        model_type=args['model_type'],
        num_point=args['data_trai_n'],
        input_data=dataset.data,
        ).to(device)

    optimizer = optim.Adam(Model.parameters(), lr=args['lr'])
    # optimizer = optim.SGD(Model.parameters(), lr=args['lr'])

    gifPloterLatentTrain = tool.GIFPloter()
    # gifPloterLatentTest = tool.GIFPloter()
    # gifPloterrecons = tool.GIFPloter()
    
    with trange(0, args['epochs'] + 1) as trange_handle:
        for epoch in trange_handle:
            # Loss.epoch = epoch
            if epoch > 0:
                loss_item = train(
                    Model=Model,
                    Loss=Loss,
                    nushadular=nushadular,
                    dataloader=dataloader,
                    optimizer=optimizer,
                    current_epoch=epoch,
                    adj=dataset.EdgeSample(args['dropedgerate']),
                    )
            else:
                loss_item=1

            if epoch % args['log_interval'] == 0:

                em_train, label = Test(
                    Model=Model,
                    Loss=Loss,
                    dataloader=dataloader,
                    optimizer=optimizer,
                    current_epoch=epoch,
                    adj=dataset.EdgeSample(0.0),
                    )
                print(em_train.shape)
                gifPloterLatentTrain.AddNewFig(
                    em_train,
                    label,
                    his_loss=None,
                    path=path,
                    graph=None,
                    link=None,
                    title_='train_epoch_em{}_{}.png'.format(
                        epoch, args['perplexity'])
                    )

                (
                    cl_acc,
                    nmi,
                    f1_macro,
                    precision_macro,
                    adjscore,
                ) = eval.TestClassifacationKMeans(
                    em_train, label
                )
                print('---------',cl_acc)

            trange_handle.set_postfix(loss=loss_item, nu=nushadular.Getnu(epoch))
            
    gifPloterLatentTrain.SaveGIF()
    return path


if __name__ == "__main__":


    # args = paramuse.GetParamCora()
    # path = main(args)
    # args = paramuse.GetParamCiteSeer()
    # path = main(args)
    args = paramuse.GetParamPubMed()
    path = main(args)