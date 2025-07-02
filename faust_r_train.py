# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:40:56 2021

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
import diffusion_net
from matching_dataset import MatchingDataset
import sys
import os
import random
# from torch_geometric.nn import knn
import scipy
import scipy.sparse.linalg as sla
from diffusion_net.utils import dist_mat, nn_search
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
from diffusion_net.utils import toNP
from diffusion_net.geometry import to_basis, from_basis
import math
import trimesh
from pyFM.mesh import TriMesh
from torch_geometric.data import Batch
from diffusion_net.data import DiffusionData
from diffusion_net.transforms import DiffusionOperatorsTransform
from diffusion_net.prism_decoder import PrismDecoder
from torch_geometric.utils import unbatch
from pyg_lib.ops import grouped_matmul
import roma
from trimesh.graph import face_adjacency
#torch.manual_seed(42)
#shrec19

def calculate_geodesic_error_1(dist_x, corr_x,  p2p, return_mean=True):
    ind21 = np.stack([corr_x, p2p], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = np.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err
# erro
#faust、scape、smal、tosca、topkids、dt4d
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


# model 
# input_features = args.input_features # one of ['xyz', 'hks']
input_features = 'hks'
k_eig = 128

# training settings
train = not args.evaluate
n_epoch = 1
lr = 1e-3

# Important paths
base_path = osp.dirname(__file__)
dataset_path = osp.join("/home/mj/ICCV/data/SCAPE")
pretrain_path = osp.join("/home/mj/ICCV/data/faust/save_models/2025_05_03__18_54_23/best.pth".format(input_features))
# model_save_path = os.path.join(dataset_path, 'saved_models','t_hk1104_faust_{}_4x128.pth'.format(input_features))


# Load the train dataset
if train:
    train_dataset = MatchingDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    now = datetime.now()
    folder_str = now.strftime("%Y_%m_%d__%H_%M_%S")
    model_save_dir=osp.join(dataset_path,'save_models',folder_str)
    model_save_dir1=osp.join(dataset_path,'save_models1',folder_str)
    diffusion_net.utils.ensure_dir_exists(model_save_dir1)
    diffusion_net.utils.ensure_dir_exists(model_save_dir)

# === Create the model

C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features



model = diffusion_net.layers.RFMNet(C_in=C_in,C_out=256)

model = model.to(device)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model1=diffusion_net.layers.PrismDecoder(259, 256)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
all_loss1=[]
def train_epoch(epoch):
    # Set model to 'train' mode
#TOSCA和SMAL分别在5和3采取迭代率下降方法
    # if epoch > 2 and epoch % 1 == 0:
    #     global lr 
    #     lr *= 0.5
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr 

    global iter
    global min_erro
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0.0
    total_num = 0
    loss_history=[]
 
    for data in tqdm(train_loader):
        for _ in range(1000):
            # Get data
            descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
                descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y=data
            
            # Move to device
            descs_x=descs_x.to(device)
            massvec_x=massvec_x.to(device)
            evals_x=evals_x.to(device)
            evecs_x=evecs_x.to(device)
            gs_x=gs_x.to(device)
            gradX_x=gradX_x.to(device) 
            gradY_x=gradY_x.to(device) #[N,N]
            elevals_x=elevals_x.to(device)
            elevecs_x=elevecs_x.to(device)

            descs_y=descs_y.to(device)
            massvec_y=massvec_y.to(device)
            evals_y=evals_y.to(device)
            evecs_y=evecs_y.to(device)
            gs_y=gs_y.to(device)
            gradX_y=gradX_y.to(device)
            gradY_y=gradY_y.to(device)
            elevals_y=elevals_y.to(device)
            elevecs_y=elevecs_y.to(device)

            # Apply the model
            loss1,feat_y,p121= model(descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
                descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y)


            feat_y=feat_y.to("cpu")
            l=feat_y.max(dim=0).values
            obj1 = r"/home/mj/h/data/SCAPE/off/"+name_y+".off"
            obj2 = r"/home/mj/h/data/SCAPE/off/"+name_x+".off"
            mesh1, mesh2 = TriMesh(obj1), TriMesh(obj2)
            mesh1_diff = trimesh.load(obj1)
            v1, f1 = np.array(mesh1.vertices), np.array(mesh1.faces)
            v1_t = torch.from_numpy(v1)
            f1_t = torch.from_numpy(f1)
            data1 = diffusion_net.data.DiffusionData(pos=v1_t, face=f1_t.T)
            diffusion_transform = diffusion_net.transforms.DiffusionOperatorsTransform(n_eig=50)  #97 compute the diffusion net operators with 97 eigenvalues
            data1 = diffusion_transform(data1)
            my_batch = Batch.from_data_list([data1])    

            my_batch.pos = my_batch.pos.unsqueeze(0) 

            mesh2_diff = trimesh.load(obj2)
            v2, f2 = np.array(mesh2.vertices), np.array(mesh2.faces)
            data2 = diffusion_net.data.DiffusionData(pos=torch.from_numpy(v2), face=torch.from_numpy(f2).T)
            diffusion_transform = diffusion_net.transforms.DiffusionOperatorsTransform(n_eig=50)  #97 compute the diffusion net operators with 97 eigenvalues
            data2 = diffusion_transform(data2)
            my_batch2 = Batch.from_data_list([data2])
            my_batch2.pos = my_batch2.pos.unsqueeze(0) 
            #######
            v2_t = torch.Tensor(v2)

            l_expanded = l.unsqueeze(0).repeat(v2_t.shape[0],1)
            
            my_batch2.x=torch.cat((v2_t,l_expanded),dim=1)
            # my_batch2=my_batch2.to("cuda")
            # decoder = model1()
            s3 = model1(my_batch2)
            get_energy_loss = diffusion_net.layers.PrismRegularizationLoss(10)
            loss2 = get_energy_loss(s3.transformed_prism, s3.rotations, s3.pos.reshape(-1, 3), s3.face)
            
            p121=p121.cpu()
            v1_remapped = v1[p121]
            loss_nn=torch.nn.MSELoss()
            loss3=loss_nn(torch.Tensor(v1_remapped),s3.features[0])
            
            loss=loss1+1*loss2+10*loss3
            # Evaluate loss
            loss1.requires_grad_(True)
            loss2.requires_grad_(True)
            loss3.requires_grad_(True)
            # p12.retain_grad()
            # clb.retain_grad()
            # evecs_x.retain_grad()
            loss.backward()

            # print(feat_x.grad)


            # track accuracy
            total_loss+=loss.item()
            loss_history.append(loss.item())
            total_num += 1
            iter+=1

            # Step the optimizer
            optimizer.step()
            optimizer.zero_grad()


            optimizer1.step()
            optimizer1.zero_grad()
            if total_num%5==0:
                print('Iterations: {:02d}, train loss: {:.4f}'.format(total_num, total_loss / total_num))
                total_loss=0.0
                #total_num=0
            if iter%100==0:
                v1_t=v1_t.to(torch.float32)
                p12=nn_search( v1_t,s3.features[0])
                data_x=sio.loadmat("/home/mj/h/data/SCAPE/dist/"+name_y+'.mat')
                dist_x=data_x["dist"]

                corr_x_file="/home/mj/h/data/SCAPE/corres/"+name_y+'.vts'
                arrays_corr_x = []
                with open(corr_x_file, 'r') as file:
                    for line in file:
                        numbers = int(line.strip())
                        arrays_corr_x.append(numbers)
                arrays_corr_x = [x-1 for x in arrays_corr_x]
                arrays_corr_x = np.array( arrays_corr_x)

                corr_y_file="/home/mj/h/data/SCAPE/corres/"+name_x+'.vts'
                arrays_corr_y = []
                with open(corr_y_file, 'r') as file:
                    for line in file:
                        numbers = int(line.strip())
                        arrays_corr_y.append(numbers)
                arrays_corr_y = [x-1 for x in arrays_corr_y]
                arrays_corr_y = np.array( arrays_corr_y)
                erro_i=calculate_geodesic_error(dist_x, arrays_corr_x,  arrays_corr_y,p12, return_mean=True)
                print(erro_i)
    #模型保存iter分别为FAUST和SCAPE是500，TOSCA是214，SMAL是992、dt4d分别是全部完整的训练数据（5292_inter、3410_intra）、TOPKIDS 650、shrec19 1892
            if iter%100==0:
                avg_erro=test()
                print(avg_erro)

                model_save_path = osp.join(model_save_dir, 'ckpt_ep{best}.pth')
                model_save_path1 = osp.join(model_save_dir1, 'ckpt_ep{best}.pth')
                if avg_erro < min_erro:
                        torch.save(model.state_dict(), model_save_path)
                        torch.save(model1.state_dict(), model_save_path1)
        
    all_loss1.append(loss_history)
    sio.savemat('SCAPE_SNK.mat', {'all_loss1': all_loss1})
    # print("Loss history saved to loss_history.mat")
def test():
    test_dataset = MatchingDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True)
    test_loader = DataLoader(test_dataset, batch_size=None)

    # file = osp.join(dataset_path, 'files_test.txt')
    # with open(file, 'r') as f:
    #     names = [line.rstrip() for line in f]

    # combinations = list(permutations(range(len(names)), 2))

    model.eval()
    with torch.no_grad():
        count=0
        erro=0
        # count_1=0
        for data in tqdm(test_loader):
            # Get data
            # Get data
            # idx1,idx2=combinations[count]
            # data_x=sio.loadmat("/home/mj/tog/checkpoints/SHREC19_r/dist/"+names[idx2]+'.mat')
            descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
                descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y=data


            # updata gs_x, gs_y
            # gs_x=diffusion_net.utils.Meyer(evals_x[n_fmap-1],Nf=Nf)(evals_x[:n_fmap]).float()
            # gs_y=diffusion_net.utils.Meyer(evals_y[n_fmap-1],Nf=Nf)(evals_y[:n_fmap]).float()


            # Move to device
            # verts_x=verts_x.to(device)
            descs_x=descs_x.to(device)
            massvec_x=massvec_x.to(device)
            evecs_x=evecs_x.to(device)
            evals_x=evals_x.to(device)
            gs_x=gs_x.to(device)
            gradX_x=gradX_x.to(device)
            gradY_x=gradY_x.to(device) #[N,N]
            # labels_x=labels_x.to(device)
            # L_x=L_x.to(device)
            elevals_x=elevals_x.to(device)
            elevecs_x=elevecs_x.to(device)

            # verts_y=verts_y.to(device)
            descs_y=descs_y.to(device)
            massvec_y=massvec_y.to(device)
            evecs_y=evecs_y.to(device)
            evals_y=evals_y.to(device)
            gs_y=gs_y.to(device)
            gradX_y=gradX_y.to(device)
            gradY_y=gradY_y.to(device)
            # labels_y=labels_y.to(device)
            # L_y=L_y.to(device)
            elevals_y=elevals_y.to(device)
            elevecs_y=elevecs_y.to(device)

            # Apply the model
            p123,feat_y= model.model_test_opt(descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
                               descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y)
            # p21= model.model_test_opt(descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,\
            #                    descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x)#TOPKIDS


            feat_y=feat_y.to("cpu")
            l=feat_y.max(dim=0).values
            obj1 = r"/home/mj/h/data/SCAPE/off/"+name_y+".off"
            obj2 = r"/home/mj/h/data/SCAPE/off/"+name_x+".off"
            mesh1, mesh2 = TriMesh(obj1), TriMesh(obj2)
            mesh1_diff = trimesh.load(obj1)
            v1, f1 = np.array(mesh1.vertices), np.array(mesh1.faces)
            v1_t = torch.from_numpy(v1)
            f1_t = torch.from_numpy(f1)
            data1 = diffusion_net.data.DiffusionData(pos=v1_t, face=f1_t.T)
            diffusion_transform = diffusion_net.transforms.DiffusionOperatorsTransform(n_eig=50)  #97 compute the diffusion net operators with 97 eigenvalues
            data1 = diffusion_transform(data1)
            my_batch = Batch.from_data_list([data1])    

            my_batch.pos = my_batch.pos.unsqueeze(0) 

            mesh2_diff = trimesh.load(obj2)
            v2, f2 = np.array(mesh2.vertices), np.array(mesh2.faces)
            data2 = diffusion_net.data.DiffusionData(pos=torch.from_numpy(v2), face=torch.from_numpy(f2).T)
            diffusion_transform = diffusion_net.transforms.DiffusionOperatorsTransform(n_eig=50)  #97 compute the diffusion net operators with 97 eigenvalues
            data2 = diffusion_transform(data2)
            my_batch2 = Batch.from_data_list([data2])
            my_batch2.pos = my_batch2.pos.unsqueeze(0) 
            #######
            v2_t = torch.Tensor(v2)

            l_expanded = l.unsqueeze(0).repeat(v2_t.shape[0],1)
            
            my_batch2.x=torch.cat((v2_t,l_expanded),dim=1)
            # my_batch2=my_batch2.to("cuda")
            # decoder = model1()
            s3 = model1(my_batch2)
            get_energy_loss = diffusion_net.layers.PrismRegularizationLoss(10)
            loss2 = get_energy_loss(s3.transformed_prism, s3.rotations, s3.pos.reshape(-1, 3), s3.face)
            
            # p121=p121.cpu()
            # v1_remapped = v1[p121]  
            v1_t=v1_t.to(torch.float32)
            p12=nn_search( v1_t,s3.features[0])
            p12=p12.cpu()
            p12=np.array(p12)
         
#SCAPE、FAUST
            # idx1,idx2=test_dataset.combinations[count]
            # name1=test_dataset.names_list[idx1]
            # name2=test_dataset.names_list[idx2]         
            data_x=sio.loadmat("/home/mj/h/data/SCAPE/dist/"+name_y+'.mat')
            dist_x=data_x["dist"]

            corr_x_file="/home/mj/h/data/SCAPE/corres/"+name_y+'.vts'
            arrays_corr_x = []
            with open(corr_x_file, 'r') as file:
               for line in file:
                 numbers = int(line.strip())
                 arrays_corr_x.append(numbers)
            arrays_corr_x = [x-1 for x in arrays_corr_x]
            arrays_corr_x = np.array( arrays_corr_x)

            corr_y_file="/home/mj/h/data/SCAPE/corres/"+name_x+'.vts'
            arrays_corr_y = []
            with open(corr_y_file, 'r') as file:
               for line in file:
                 numbers = int(line.strip())
                 arrays_corr_y.append(numbers)
            arrays_corr_y = [x-1 for x in arrays_corr_y]
            arrays_corr_y = np.array( arrays_corr_y)
            erro_i=calculate_geodesic_error(dist_x, arrays_corr_x,  arrays_corr_y,p12, return_mean=True)
# TOPKIDS
            # idx1,idx2=test_dataset.combinations[count]
            # name1=test_dataset.names_list[idx1]
            # name2=test_dataset.names_list[idx2]  

            # # idx1,idx2=combinations[count]
            # data_x=sio.loadmat("/home/mj/h/data/TOPKIDS/dist/"+name1+'.mat')
            # dist_x=data_x["dist"]

            # corr_x_file="//home/mj/h/data/TOPKIDS/corres/"+name2+'_ref.vts'
            # arrays_corr_x = []
            # with open(corr_x_file, 'r') as file:
            #    for line in file:
            #      numbers = int(line.strip())
            #      arrays_corr_x.append(numbers)
            # arrays_corr_x = [x-1  for x in arrays_corr_x]
            # arrays_corr_x = np.array( arrays_corr_x)
            # arrays_corr_y=torch.arange(0, len(arrays_corr_x)).long()
            # erro_i=calculate_geodesic_error(dist_x, arrays_corr_x,  arrays_corr_y,p21, return_mean=True)
#TOSCA
            # idx1,idx2=test_dataset.combinations[count]
            # name1=test_dataset.names_list[idx1]
            # name2=test_dataset.names_list[idx2]  

            # # idx1,idx2=combinations[count]
            # data_x=sio.loadmat("/home/mj/tog/TOSCA_r/dist/"+name2+'.mat')
            # dist_x=data_x["dist"]

            # corr_x_file="/home/mj/tog/TOSCA_r/corres/"+name2+'.vts'
            # arrays_corr_x = []
            # with open(corr_x_file, 'r') as file:
            #    for line in file:
            #      numbers = int(line.strip())
            #      arrays_corr_x.append(numbers)
            # arrays_corr_x = [x-1  for x in arrays_corr_x]
            # arrays_corr_x = np.array( arrays_corr_x)
          
            # corr_y_file="/home/mj/tog/TOSCA_r/corres/"+name1+'.vts'
            # arrays_corr_y = []
            # with open(corr_y_file, 'r') as file:
            #    for line in file:
            #      numbers = int(line.strip())
            #      arrays_corr_y.append(numbers)
            # arrays_corr_y = [x-1  for x in arrays_corr_y]
            # arrays_corr_y = np.array( arrays_corr_y)
            # erro_i=calculate_geodesic_error(dist_x, arrays_corr_x,  arrays_corr_y,p12, return_mean=True)
#DT4D——intraclass
            # idx1,idx2=test_dataset.combinations[count]
            # name1=test_dataset.names_list[idx1]
            # name2=test_dataset.names_list[idx2]  

            # # idx1,idx2=combinations[count]
            # data_x=sio.loadmat("/home/mj/h/data/DT4D_r/dist/"+name2+'.mat')
            # dist_x=data_x["dist"]

            # corr_x_file="/home/mj/h/data/DT4D_r/corres_intra/"+name2+'.vts'
            # arrays_corr_x = []
            # with open(corr_x_file, 'r') as file:
            #    for line in file:
            #      numbers = int(line.strip())
            #      arrays_corr_x.append(numbers)
            # arrays_corr_x = [x-1  for x in arrays_corr_x]
            # arrays_corr_x = np.array( arrays_corr_x)
          

            # corr_y_file="/home/mj/h/data/DT4D_r/corres_intra/"+name1+'.vts'
            # arrays_corr_y = []
            # with open(corr_y_file, 'r') as file:
            #    for line in file:
            #      numbers = int(line.strip())
            #      arrays_corr_y.append(numbers)
            # arrays_corr_y = [x-1  for x in arrays_corr_y]
            # arrays_corr_y = np.array( arrays_corr_y)

            # erro_i=calculate_geodesic_error(dist_x, arrays_corr_x,  arrays_corr_y,p12, return_mean=True)
# DT4D_inter-class
            # idx1,idx2=test_dataset.combinations[count]
            # name1=test_dataset.names_list[idx1]
            # name2=test_dataset.names_list[idx2]  

            # first_cat = test_dataset.off_files[idx1].split('/')[-2]
            # second_cat = test_dataset.off_files[idx2].split('/')[-2]
            # corr = np.loadtxt(os.path.join("/home/mj/h/data/DT4D_r/corres/cross_category_corres",
            #                                f'{first_cat}_{second_cat}.vts'), dtype=np.int32) -1

            # data_x=sio.loadmat("/home/mj/h/data/DT4D_r/dist/"+name2+'.mat')
            # dist_x=data_x["dist"]

            # corr_x_file="/home/mj/h/data/DT4D_r/corres_intra/"+name2+'.vts'
            # arrays_corr_x = []
            # with open(corr_x_file, 'r') as file:
            #    for line in file:
            #      numbers = int(line.strip())
            #      arrays_corr_x.append(numbers)
            # arrays_corr_x = [x-1  for x in arrays_corr_x]
            # arrays_corr_x = np.array( arrays_corr_x)
            # arrays_corr_x =arrays_corr_x [corr]

            # corr_y_file="/home/mj/h/data/DT4D_r/corres_intra/"+name1+'.vts'
            # arrays_corr_y = []
            # with open(corr_y_file, 'r') as file:
            #    for line in file:
            #      numbers = int(line.strip())
            #      arrays_corr_y.append(numbers)
            # arrays_corr_y = [x-1  for x in arrays_corr_y]
            # arrays_corr_y = np.array( arrays_corr_y)

            # erro_i=calculate_geodesic_error(dist_x, arrays_corr_x,  arrays_corr_y,p12, return_mean=True)

# SHREC19
            # idx1,idx2=test_dataset.combinations[count]
            # name1=test_dataset.names_list[idx1]
            # name2=test_dataset.names_list[idx2]              
            # data_x=sio.loadmat("/home/mj/tog/checkpoints/SHREC19/dist/"+name_y+'.mat')
            # dist_x=data_x["dist"]

            # corr_x_file="/home/mj/tog/checkpoints/SHREC19/corres/"+name_x+"_"+name_y+'.map'

            # count+=1
            # if not os.path.exists(corr_x_file):
            #     print("文件不存在:", corr_x_file)
            #     continue
            # arrays_corr_x = []
            # with open(corr_x_file, 'r') as file:
            #     for line in file:
            #         numbers = int(line.strip())
            #         arrays_corr_x.append(numbers)
            # arrays_corr_x = [x-1  for x in arrays_corr_x]
            # arrays_corr_x = np.array( arrays_corr_x)
            # erro_i=calculate_geodesic_error_1(dist_x, arrays_corr_x,p12, return_mean=True)
            # print(names[idx1]+"_"+names[idx2]+erro_i)
            # with open("/home/mj/DeepWaveletFMNet/data/scape_5k/S_SHREC_30","a")as f:
            #       f.write(f"{names[idx1]+"_"+names[idx2]+":"+erro_i}\r\n")
            count+=1
            erro=erro+erro_i

        avg_erro=erro/count
        # with open("/home/mj/h/data/DT4D_r/inter","a")as f:
        #     f.write(f"{avg_erro}\r\n")
        return avg_erro

if train:
    print("Training...")
    iter=0
    min_erro=100
    # min_loss=1e10
    for epoch in range(n_epoch):
        # torch.cuda.empty_cache()
        # start_time = time.time()
        train_epoch(epoch)
# if train:
#     print("Training...")

#     for epoch in range(n_epoch):
#         train_acc = train_epoch(epoch)
        
#         model_save_path=osp.join(model_save_dir,'ckpt_ep{}.pth'.format(n_epoch))
#         torch.save(model.state_dict(), model_save_path)

#     print(" ==> saving last model to " + model_save_path)
#     torch.save(model.state_dict(), model_save_path)
 

# Test
# test_dataset = MatchingDataset("/home/mj/ICCV/data/smal", train=False, k_eig=k_eig, use_cache=True)
# test_loader = DataLoader(test_dataset, batch_size=None)
# # results_dir = osp.join("/home/mj/ICCV/data/DT4D/T")
# # file = osp.join(dataset_path, 'files_test.txt')
# # with open(file, 'r') as f:
# #     names = [line.rstrip() for line in f]

# # combinations = list(permutations(range(len(names)), 2))

# model.eval()
# with torch.no_grad():
#     count=0
#     erro=0
#     # count_1=0
#     for data in tqdm(test_loader):
#         # Get data
#         # Get data
#         # idx1,idx2=combinations[count]
#         # data_x=sio.loadmat("/home/mj/tog/checkpoints/SHREC19_r/dist/"+names[idx2]+'.mat')
#             # Get data
#             # Get data
#             # idx1,idx2=combinations[count]
#             # data_x=sio.loadmat("/home/mj/tog/checkpoints/SHREC19_r/dist/"+names[idx2]+'.mat')
#             descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
#                 descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y=data


#             # updata gs_x, gs_y
#             # gs_x=diffusion_net.utils.Meyer(evals_x[n_fmap-1],Nf=Nf)(evals_x[:n_fmap]).float()
#             # gs_y=diffusion_net.utils.Meyer(evals_y[n_fmap-1],Nf=Nf)(evals_y[:n_fmap]).float()


#             # Move to device
#             # verts_x=verts_x.to(device)
#             descs_x=descs_x.to(device)
#             massvec_x=massvec_x.to(device)
#             evecs_x=evecs_x.to(device)
#             evals_x=evals_x.to(device)
#             gs_x=gs_x.to(device)
#             gradX_x=gradX_x.to(device)
#             gradY_x=gradY_x.to(device) #[N,N]
#             # labels_x=labels_x.to(device)
#             # L_x=L_x.to(device)
#             elevals_x=elevals_x.to(device)
#             elevecs_x=elevecs_x.to(device)

#             # verts_y=verts_y.to(device)
#             descs_y=descs_y.to(device)
#             massvec_y=massvec_y.to(device)
#             evecs_y=evecs_y.to(device)
#             evals_y=evals_y.to(device)
#             gs_y=gs_y.to(device)
#             gradX_y=gradX_y.to(device)
#             gradY_y=gradY_y.to(device)
#             # labels_y=labels_y.to(device)
#             # L_y=L_y.to(device)
#             elevals_y=elevals_y.to(device)
#             elevecs_y=elevecs_y.to(device)

#             # Apply the model
#             p12= model.model_test_opt(descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
#                                descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y)
#             # p21= model.model_test_opt(descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,\
#             #                    descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x)#TOPKIDS
# # 后处理


        



#             # feat_y=feat_y.to("cpu")
#             # l=feat_y.max(dim=0).values
#             # obj1 = r"/home/mj/h/data/FAUST/off/"+name_y+".off"
#             # obj2 = r"/home/mj/h/data/FAUST/off/"+name_x+".off"
#             # mesh1, mesh2 = TriMesh(obj1), TriMesh(obj2)
#             # mesh1_diff = trimesh.load(obj1)
#             # v1, f1 = np.array(mesh1.vertices), np.array(mesh1.faces)
#             # v1_t = torch.from_numpy(v1)
#             # f1_t = torch.from_numpy(f1)
#             # data1 = diffusion_net.data.DiffusionData(pos=v1_t, face=f1_t.T)
#             # diffusion_transform = diffusion_net.transforms.DiffusionOperatorsTransform(n_eig=50)  #97 compute the diffusion net operators with 97 eigenvalues
#             # data1 = diffusion_transform(data1)
#             # my_batch = Batch.from_data_list([data1])    

#             # my_batch.pos = my_batch.pos.unsqueeze(0) 

#             # mesh2_diff = trimesh.load(obj2)
#             # v2, f2 = np.array(mesh2.vertices), np.array(mesh2.faces)
#             # data2 = diffusion_net.data.DiffusionData(pos=torch.from_numpy(v2), face=torch.from_numpy(f2).T)
#             # diffusion_transform = diffusion_net.transforms.DiffusionOperatorsTransform(n_eig=50)  #97 compute the diffusion net operators with 97 eigenvalues
#             # data2 = diffusion_transform(data2)
#             # my_batch2 = Batch.from_data_list([data2])
#             # my_batch2.pos = my_batch2.pos.unsqueeze(0) 
#             # #######
#             # v2_t = torch.Tensor(v2)

#             # l_expanded = l.unsqueeze(0).repeat(v2_t.shape[0],1)
            
#             # my_batch2.x=torch.cat((v2_t,l_expanded),dim=1)
#             # # my_batch2=my_batch2.to("cuda")
#             # # decoder = model1()
#             # s3 = model1(my_batch2)
#             # get_energy_loss = diffusion_net.layers.PrismRegularizationLoss(10)
#             # loss2 = get_energy_loss(s3.transformed_prism, s3.rotations, s3.pos.reshape(-1, 3), s3.face)
            
#             # p121=p121.cpu()
#             # v1_remapped = v1[p121]
#             # p12=p121
#             p12=p12.cpu()
#             p12=np.array(p12)
# #SCAPE、FAUST
#             # idx1,idx2=test_dataset.combinations[count]
#             # name1=test_dataset.names_list[idx1]
#             # name2=test_dataset.names_list[idx2]         
#             data_x=sio.loadmat("/home/mj/h/data/SMAL_r/dist/"+name_y+'.mat')
#             dist_x=data_x["dist"]

#             corr_x_file="/home/mj/h/data/SMAL_r/corres/"+name_y+'.vts'
#             arrays_corr_x = []
#             with open(corr_x_file, 'r') as file:
#                for line in file:
#                  numbers = int(line.strip())
#                  arrays_corr_x.append(numbers)
#             arrays_corr_x = [x-1 for x in arrays_corr_x]
#             arrays_corr_x = np.array( arrays_corr_x)

#             corr_y_file="/home/mj/h/data/SMAL_r/corres/"+name_x+'.vts'
#             arrays_corr_y = []
#             with open(corr_y_file, 'r') as file:
#                for line in file:
#                  numbers = int(line.strip())
#                  arrays_corr_y.append(numbers)
#             arrays_corr_y = [x-1 for x in arrays_corr_y]
#             arrays_corr_y = np.array( arrays_corr_y)
#             erro_i=calculate_geodesic_error(dist_x, arrays_corr_x,  arrays_corr_y,p12, return_mean=True)
# # TOPKIDS
#             # idx1,idx2=test_dataset.combinations[count]
#             # name1=test_dataset.names_list[idx1]
#             # name2=test_dataset.names_list[idx2]  

#             # # idx1,idx2=combinations[count]
#             # data_x=sio.loadmat("/home/mj/h/data/TOPKIDS/dist/"+name1+'.mat')
#             # dist_x=data_x["dist"]

#             # corr_x_file="//home/mj/h/data/TOPKIDS/corres/"+name2+'_ref.vts'
#             # arrays_corr_x = []
#             # with open(corr_x_file, 'r') as file:
#             #    for line in file:
#             #      numbers = int(line.strip())
#             #      arrays_corr_x.append(numbers)
#             # arrays_corr_x = [x-1  for x in arrays_corr_x]
#             # arrays_corr_x = np.array( arrays_corr_x)
#             # arrays_corr_y=torch.arange(0, len(arrays_corr_x)).long()
#             # erro_i=calculate_geodesic_error(dist_x, arrays_corr_x,  arrays_corr_y,p21, return_mean=True)
# #TOSCA
#             # idx1,idx2=test_dataset.combinations[count]
#             # name1=test_dataset.names_list[idx1]
#             # name2=test_dataset.names_list[idx2]  

#             # # idx1,idx2=combinations[count]
#             # data_x=sio.loadmat("/home/mj/tog/TOSCA_r/dist/"+name2+'.mat')
#             # dist_x=data_x["dist"]

#             # corr_x_file="/home/mj/tog/TOSCA_r/corres/"+name2+'.vts'
#             # arrays_corr_x = []
#             # with open(corr_x_file, 'r') as file:
#             #    for line in file:
#             #      numbers = int(line.strip())
#             #      arrays_corr_x.append(numbers)
#             # arrays_corr_x = [x-1  for x in arrays_corr_x]
#             # arrays_corr_x = np.array( arrays_corr_x)
          
#             # corr_y_file="/home/mj/tog/TOSCA_r/corres/"+name1+'.vts'
#             # arrays_corr_y = []
#             # with open(corr_y_file, 'r') as file:
#             #    for line in file:
#             #      numbers = int(line.strip())
#             #      arrays_corr_y.append(numbers)
#             # arrays_corr_y = [x-1  for x in arrays_corr_y]
#             # arrays_corr_y = np.array( arrays_corr_y)
#             # erro_i=calculate_geodesic_error(dist_x, arrays_corr_x,  arrays_corr_y,p12, return_mean=True)
# #DT4D——intraclass
#             # idx1,idx2=test_dataset.combinations[count]
#             # name1=test_dataset.names_list[idx1]
#             # name2=test_dataset.names_list[idx2]  

#             # # idx1,idx2=combinations[count]
#             # data_x=sio.loadmat("/home/mj/h/data/DT4D_r/dist/"+name2+'.mat')
#             # dist_x=data_x["dist"]

#             # corr_x_file="/home/mj/h/data/DT4D_r/corres_intra/"+name2+'.vts'
#             # arrays_corr_x = []
#             # with open(corr_x_file, 'r') as file:
#             #    for line in file:
#             #      numbers = int(line.strip())
#             #      arrays_corr_x.append(numbers)
#             # arrays_corr_x = [x-1  for x in arrays_corr_x]
#             # arrays_corr_x = np.array( arrays_corr_x)
          

#             # corr_y_file="/home/mj/h/data/DT4D_r/corres_intra/"+name1+'.vts'
#             # arrays_corr_y = []
#             # with open(corr_y_file, 'r') as file:
#             #    for line in file:
#             #      numbers = int(line.strip())
#             #      arrays_corr_y.append(numbers)
#             # arrays_corr_y = [x-1  for x in arrays_corr_y]
#             # arrays_corr_y = np.array( arrays_corr_y)

#             # erro_i=calculate_geodesic_error(dist_x, arrays_corr_x,  arrays_corr_y,p12, return_mean=True)
# # DT4D_inter-class
#             # idx1,idx2=test_dataset.combinations[count]
#             # name1=test_dataset.names_list[idx1]
#             # name2=test_dataset.names_list[idx2]  

#             # first_cat = test_dataset.off_files[idx1].split('/')[-2]
#             # second_cat = test_dataset.off_files[idx2].split('/')[-2]
#             # corr = np.loadtxt(os.path.join("/home/mj/h/data/DT4D_r/corres/cross_category_corres",
#             #                                f'{first_cat}_{second_cat}.vts'), dtype=np.int32) -1

#             # data_x=sio.loadmat("/home/mj/h/data/DT4D_r/dist/"+name2+'.mat')
#             # dist_x=data_x["dist"]

#             # corr_x_file="/home/mj/h/data/DT4D_r/corres_intra/"+name2+'.vts'
#             # arrays_corr_x = []
#             # with open(corr_x_file, 'r') as file:
#             #    for line in file:
#             #      numbers = int(line.strip())
#             #      arrays_corr_x.append(numbers)
#             # arrays_corr_x = [x-1  for x in arrays_corr_x]
#             # arrays_corr_x = np.array( arrays_corr_x)
#             # arrays_corr_x =arrays_corr_x [corr]

#             # corr_y_file="/home/mj/h/data/DT4D_r/corres_intra/"+name1+'.vts'
#             # arrays_corr_y = []
#             # with open(corr_y_file, 'r') as file:
#             #    for line in file:
#             #      numbers = int(line.strip())
#             #      arrays_corr_y.append(numbers)
#             # arrays_corr_y = [x-1  for x in arrays_corr_y]
#             # arrays_corr_y = np.array( arrays_corr_y)

#             # erro_i=calculate_geodesic_error(dist_x, arrays_corr_x,  arrays_corr_y,p12, return_mean=True)

# # SHREC19
#             # idx1,idx2=test_dataset.combinations[count]
#             # name1=test_dataset.names_list[idx1]
#             # name2=test_dataset.names_list[idx2]              
#             # data_x=sio.loadmat("/home/mj/tog/checkpoints/SHREC19/dist/"+name_y+'.mat')
#             # dist_x=data_x["dist"]

#             # corr_x_file="/home/mj/tog/checkpoints/SHREC19/corres/"+name_x+"_"+name_y+'.map'

#             # count+=1
#             # if not os.path.exists(corr_x_file):
#             #     print("文件不存在:", corr_x_file)
#             #     continue
#             # arrays_corr_x = []
#             # with open(corr_x_file, 'r') as file:
#             #     for line in file:
#             #         numbers = int(line.strip())
#             #         arrays_corr_x.append(numbers)
#             # arrays_corr_x = [x-1  for x in arrays_corr_x]
#             # arrays_corr_x = np.array( arrays_corr_x)
#             # erro_i=calculate_geodesic_error_1(dist_x, arrays_corr_x,p12, return_mean=True)
#             # print(names[idx1]+"_"+names[idx2]+erro_i)
#             # with open("/home/mj/DeepWaveletFMNet/data/scape_5k/S_SHREC_30","a")as f:
#             #       f.write(f"{names[idx1]+"_"+names[idx2]+":"+erro_i}\r\n")
#             count+=1
#             erro=erro+erro_i

#     avg_erro=erro/count
#     print(avg_erro)