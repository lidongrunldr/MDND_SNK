# -*- coding: utf-8 -*-
"""
Created on Sun May 22 10:00:14 2022

@author: Michael
"""

import os
import sys
from itertools import permutations
import os.path as osp
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import scipy.io as sio

sys.path.append(osp.join(os.getcwd(),'src'))
import src.diffusion_net as diffusion_net

from tosca_r_dataset import MatchingDataset

from src.diffusion_net.utils import nn_search

def calculate_geodesic_error(dist_x, corr_x, corr_y, p2p, return_mean=True):
    ind21 = np.stack([corr_x, p2p[corr_y]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = np.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err
# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
args = parser.parse_args()


# system things
device = torch.device('cuda:0')
dtype = torch.float32

# problem/dataset things
# n_class = 8

# model 
# input_features = args.input_features # one of ['xyz', 'hks']
input_features = 'WKS'
k_eig = 128
Nf=6
n_fmap=100
# training settings
train =  args.evaluate
# train = args.evaluate
n_epoch = 20
lr = 1e-3
decay_every = 1
decay_rate = 0.5

# Important paths
base_path = osp.dirname(__file__)
dataset_path = osp.join(base_path, 'data','tosca')
pretrain_path = osp.join("/home/mj/DeepWaveletFMNet/data/tosca/best.pth")
# model_save_path = os.path.join(dataset_path, 'saved_models','t_hk1104_faust_{}_4x128.pth'.format(input_features))


# Load the train dataset
if train:
    train_dataset = MatchingDataset(dataset_path, train=True, k_eig=k_eig, Nf=Nf,use_cache=True)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    now = datetime.now()
    folder_str = now.strftime("%Y_%m_%d__%H_%M_%S")
    model_save_dir=osp.join(dataset_path,'save_models',folder_str)
    diffusion_net.utils.ensure_dir_exists(model_save_dir)

# === Create the model

C_in={'xyz':3, 'hks':16, 'WKS':128}[input_features] # dimension of input features

# model = diffusion_net.layers.DiffusionNet(C_in=C_in,
#                                           C_out=C_in,
#                                           C_width=128, 
#                                           N_block=4, 
#                                           last_activation=None,
#                                           outputs_at='vertices', 
#                                           dropout=True)

model = diffusion_net.layers.SSWFMNet(C_in=C_in,C_out=256,n_fmap=n_fmap)

model = model.to(device)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_iter_loss=[]
def train_epoch(epoch):

    # Implement lr decay
    if epoch > 2 and epoch % decay_every == 0:
        global lr 
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 

    global iter
    global min_erro
    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0.0
    total_num = 0
    for data in tqdm(train_loader):

        # Get data
        descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y=data

        gs_x = diffusion_net.utils.Meyer(evals_x[n_fmap - 1], Nf=Nf)(evals_x[:n_fmap]).float()
        gs_y = diffusion_net.utils.Meyer(evals_y[n_fmap - 1], Nf=Nf)(evals_y[:n_fmap]).float()        
        # Move to device
        descs_x=descs_x.to(device)
        massvec_x=massvec_x.to(device)
        evals_x=evals_x.to(device)
        evecs_x=evecs_x.to(device)
        gs_x=gs_x.to(device)
        gradX_x=gradX_x.to(device) 
        gradY_x=gradY_x.to(device) #[N,N]

        descs_y=descs_y.to(device)
        massvec_y=massvec_y.to(device)
        evals_y=evals_y.to(device)
        evecs_y=evecs_y.to(device)
        gs_y=gs_y.to(device)
        gradX_y=gradX_y.to(device)
        gradY_y=gradY_y.to(device)

        
        # Apply the model
        loss = model(descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y)

        # Evaluate loss
        loss.backward()
        
        # track accuracy
        total_loss+=loss.item()
        total_num += 1
        iter += 1
        train_iter_loss.append(loss.item())
        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

        if total_num%100==0:
            print('Iterations: {:02d}, train loss: {:.4f}'.format(total_num, total_loss / total_num))
            total_loss=0.0
            total_num=0

        if iter%214==0:
            avg_erro=test()
            print(avg_erro)
            model_save_path = osp.join(model_save_dir, 'ckpt_ep{best}.pth')
            if avg_erro < min_erro:
                    torch.save(model.state_dict(), model_save_path)
def test():
    test_dataset = MatchingDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True)
    test_loader = DataLoader(test_dataset, batch_size=None)

    results_dir = osp.join(model_save_dir, 'WKS_SSWFMNet_results')
    diffusion_net.utils.ensure_dir_exists(results_dir)
    sio.savemat(osp.join(results_dir, 'train_iter_loss.mat'),{'train_iter_loss': np.array(train_iter_loss).astype(np.float32)})
    
    file = osp.join(dataset_path, 'files_test.txt')
    with open(file, 'r') as f:
        names = [line.rstrip() for line in f]

    combinations = list(permutations(range(len(names)), 2))

    model.eval()
    with torch.no_grad():
        count = 0
        erro = 0
        # min_erro=1
        for data in tqdm(test_loader):
            # Get data
            # Get data
            descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
                descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y=data

            gs_x = diffusion_net.utils.Meyer(evals_x[n_fmap - 1], Nf=Nf)(evals_x[:n_fmap]).float()
            gs_y = diffusion_net.utils.Meyer(evals_y[n_fmap - 1], Nf=Nf)(evals_y[:n_fmap]).float()           
            # Move to device
            descs_x=descs_x.to(device)
            massvec_x=massvec_x.to(device)
            evals_x=evals_x.to(device)
            evecs_x=evecs_x.to(device)
            gs_x=gs_x.to(device)
            gradX_x=gradX_x.to(device) 
            gradY_x=gradY_x.to(device) #[N,N]

            descs_y=descs_y.to(device)
            massvec_y=massvec_y.to(device)
            evals_y=evals_y.to(device)
            evecs_y=evecs_y.to(device)
            gs_y=gs_y.to(device)
            gradX_y=gradX_y.to(device)
            gradY_y=gradY_y.to(device)

            # Apply the model
            p12= model.model_test(descs_x, massvec_x, evals_x, evecs_x, gs_x, gradX_x, gradY_x,\
                                    descs_y, massvec_y, evals_y, evecs_y, gs_y, gradX_y, gradY_y)

            # kx,ky= C_raw21.size()
            # T_12 = nn_search(evecs_y[:, :ky] @ C_raw21.t(), evecs_x[:, :kx])
            p12 =p12 .cpu()
            p12 = np.array(p12)
            # p21 =p21 .cpu()
            # p21 = np.array(p21)
            # T_12 =T_12 .cpu()
            # T_12 = np.array(T_12)


            # idx1, idx2 = combinations[count]
            idx1,idx2=test_dataset.combinations[count]
            name1=test_dataset.names_list[idx1]
            name2=test_dataset.names_list[idx2]            
            # if epoch==9:
            #     T21_name ="/home/mj/smoothFM-main/T/{}_{}".format(names[idx2],names[idx1])
            #     # T21_name ="/home/mj/smoothFM-main/T/"+names[idx2]+"_"+names[idx1]
            #     T12_name ="/home/mj/smoothFM-main/T/{}_{}".format(names[idx1],names[idx2])
            #     # T12_name ="/home/mj/smoothFM-main/T/"+names[idx1]+"_"+names[idx2]
            #     np.save(T12_name, p12)
            #     np.save(T21_name, p21)

            data_x = sio.loadmat( "/home/mj/tog/TOSCA_r/dist/" +name2 + '.mat')
            dist_x = data_x["dist"]
            corr_x_file = "/home/mj/tog/TOSCA_r/corres/" +name2 + '.vts'
            arrays_corr_x = []
            with open(corr_x_file, 'r') as file:
                for line in file:
                    numbers = int(line.strip())
                    arrays_corr_x.append(numbers)
            arrays_corr_x = [x-1  for x in arrays_corr_x]
            arrays_corr_x = np.array(arrays_corr_x)
            corr_y_file = "/home/mj/tog/TOSCA_r/corres/" +name1 + '.vts'
            arrays_corr_y = []
            with open(corr_y_file, 'r') as file:
                for line in file:
                    numbers = int(line.strip())
                    arrays_corr_y.append(numbers)
            arrays_corr_y = [x-1 for x in arrays_corr_y]
            arrays_corr_y = np.array(arrays_corr_y)
            # arrays_corr_x = arrays_corr_x.astype(np.float64)
            # arrays_corr_y = arrays_corr_y.astype(np.float64)
            # dist_x=dist_x.astype(np.float32)
            erro_i = calculate_geodesic_error(dist_x, arrays_corr_x, arrays_corr_y,p12, return_mean=True)

            count += 1
            erro = erro + erro_i

        avg_erro = erro / count
        with open("/home/mj/DeepWaveletFMNet/data/tosca/com_10","a")as f:
            f.write(f"{avg_erro}\r\n")
        return avg_erro    
                   
if train:
    print("Training...")
    iter=0
    min_erro=1
    for epoch in range(n_epoch):
        # torch.cuda.empty_cache()
        train_epoch(epoch)
    

# def axio_MWP(massvec_x,evecs_x,gs_x,evecs_y,gs_y,T,num_iter=5):
#     # input: 
#     #   massvec_x: [M,]
#     #   evecs_x/y: [M/N,Kx/Ky]
#     #   gs_x/y: [Nf,Kx/Ky]
#     #   T: [M,]
#     gs_x=gs_x.unsqueeze(-1) # [Nf,Kx]->[Nf,Kx,1]
#     gs_y=gs_y.unsqueeze(-1) # [Nf,Ky]->[Nf,Ky,1]
#     Nf=gs_x.size(0)
    
#     for it in range(num_iter):
#         C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(evecs_y[T,:]))
#         C_new=torch.zeros_like(C)
        
#         for s in range(Nf):
#             C_new+=gs_x[s]*C*gs_y[s].transpose(-2,-1)
        
#         T=nearest_neighbor(evecs_x,evecs_y@C.t())
    
#     return C_new, T
        

# Test
# Load the test dataset
test_dataset = MatchingDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True)
test_loader = DataLoader(test_dataset, batch_size=None,shuffle=False)

# model_save_dir=osp.join(dataset_path, 'save_models','2022_05_22__15_01_29')

# results_dir=osp.join(model_save_dir,'hks_results')
# diffusion_net.utils.ensure_dir_exists(results_dir)

# mwp_refined_results_dir=osp.join(results_dir,'mwp_refined_results')
# if not osp.isdir(mwp_refined_results_dir):
#         os.makedirs(mwp_refined_results_dir)

model.eval()
with torch.no_grad():
    count=0
    for data in tqdm(test_loader):
        descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y=data
        
        # Move to device
        descs_x=descs_x.to(device)
        massvec_x=massvec_x.to(device)
        evals_x=evals_x.to(device)
        evecs_x=evecs_x.to(device)
        gs_x=gs_x.to(device)
        gradX_x=gradX_x.to(device)
        gradY_x=gradY_x.to(device)

        descs_y=descs_y.to(device)
        massvec_y=massvec_y.to(device)
        evals_y=evals_y.to(device)
        evecs_y=evecs_y.to(device)
        gs_y=gs_y.to(device)
        gradX_y=gradX_y.to(device)
        gradY_y=gradY_y.to(device)

    
        # Apply the model
        loss, C = model(descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y)
        
        # model.feat_correspondences(evecs_x,evecs_y@C.t())
        # T=torch.argmax(model.p,dim=1)
        # T=nn_search(evecs_y@C.t(),evecs_x)
        
        idx1,idx2=test_dataset.combinations[count]
        name1=test_dataset.names_list[idx1]
        name2=test_dataset.names_list[idx2]
        count+=1
        data_x = sio.loadmat( "/home/mj/tog/TOSCA_r/dist/" +name2 + '.mat')
        dist_x = data_x["dist"]
        corr_x_file = "/home/mj/tog/TOSCA_r/corres/" +name2 + '.vts'
        arrays_corr_x = []
        with open(corr_x_file, 'r') as file:
            for line in file:
                numbers = int(line.strip())
                arrays_corr_x.append(numbers)
        arrays_corr_x = [x-1  for x in arrays_corr_x]
        arrays_corr_x = np.array(arrays_corr_x)
        corr_y_file = "/home/mj/tog/TOSCA_r/corres/" +name1 + '.vts'
        arrays_corr_y = []
        with open(corr_y_file, 'r') as file:
            for line in file:
                numbers = int(line.strip())
                arrays_corr_y.append(numbers)
        arrays_corr_y = [x-1 for x in arrays_corr_y]
        arrays_corr_y = np.array(arrays_corr_y)
        # arrays_corr_x = arrays_corr_x.astype(np.float64)
        # arrays_corr_y = arrays_corr_y.astype(np.float64)
        # dist_x=dist_x.astype(np.float32)
        erro_i = calculate_geodesic_error(dist_x, arrays_corr_x, arrays_corr_y,p12, return_mean=True)

        count += 1
        erro = erro + erro_i

    # avg_erro = erro / count
    with open("/home/mj/DeepWaveletFMNet/data/tosca/New File","a")as f:
        f.write(f"{name1+name2+erro_i}\r\n")
    # print(avg_erro )
        results_path=osp.join(results_dir,name1+'_'+name2+'.mat')
        sio.savemat(results_path, {'C':C.to('cpu').numpy().astype(np.float32),
                                   'T':T.to('cpu').numpy().astype(np.int64)+1}) # T: convert to matlab index
        
        # C,T=axioMWP(massvec_x,evecs_x,gs_x,evecs_y,gs_y,T,num_iter=10)

        # results_path=osp.join(mwp_refined_results_dir,name1+'_'+name2+'.mat')
        # sio.savemat(results_path, {'C':C.to('cpu').numpy().astype(np.float32),
        #                             'T':T.to('cpu').numpy().astype(np.int64)+1}) # T: convert to matlab index


