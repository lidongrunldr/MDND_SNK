import os
import sys
from itertools import permutations

import numpy as np
import scipy.io as sio

import potpourri3d as pp3d
from potpourri3d.mesh import vertex_areas

import torch
from torch.utils.data import Dataset

import os.path as osp

sys.path.append(osp.join(os.getcwd(),'src'))
import diffusion_net

# Meyer filters
class Meyer(object):
    def __init__(self, lmax, Nf=6, scales=None):

        self.Nf=Nf

        if scales is None:
            scales = (4./(3 * lmax)) * np.power(2., np.arange(Nf-2, -1, -1))

        if len(scales) != Nf - 1:
            raise ValueError('len(scales) should be Nf-1.')

        self.g = [lambda x: kernel(scales[0] * x, 'scaling_function')]

        for i in range(Nf - 1):
            self.g.append(lambda x, i=i: kernel(scales[i] * x, 'wavelet'))

        def kernel(x, kernel_type):
            r"""
            Evaluates Meyer function and scaling function

            * meyer wavelet kernel: supported on [2/3,8/3]
            * meyer scaling function kernel: supported on [0,4/3]
            """

            x = np.asarray(x)

            l1 = 2/3.
            l2 = 4/3.  # 2*l1
            l3 = 8/3.  # 4*l1

            def v(x):
                return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)

            r1ind = (x < l1)
            r2ind = (x >= l1) * (x < l2)
            r3ind = (x >= l2) * (x < l3)

            # as we initialize r with zero, computed function will implicitly
            # be zero for all x not in one of the three regions defined above
            r = np.zeros(x.shape)
            if kernel_type == 'scaling_function':
                r[r1ind] = 1
                r[r2ind] = np.cos((np.pi/2) * v(np.abs(x[r2ind])/l1 - 1))
            elif kernel_type == 'wavelet':
                r[r2ind] = np.sin((np.pi/2) * v(np.abs(x[r2ind])/l1 - 1))
                r[r3ind] = np.cos((np.pi/2) * v(np.abs(x[r3ind])/l2 - 1))
            else:
                raise ValueError('Unknown kernel type {}'.format(kernel_type))

            return r


    def __call__(self, evals):
        # input:
        #   evals: [K,], pytorch tensor
        # output: 
        #   gs: [Nf,K,], pytorch tensor
        evals=evals.numpy()
        gs=np.expand_dims(self.g[0](evals),0)

        for s in range(1, self.Nf):
            gs=np.concatenate((gs,np.expand_dims(self.g[s](evals),0)),0)
        
        return torch.from_numpy(gs.astype(np.float32))

class MatchingDataset(Dataset):
    def __init__(self,root_dir,train,k_eig=128,Nf=6,use_cache=True):
        super().__init__()

        self.root_dir=root_dir
        self.train=train
        self.k_eig=k_eig
        self.Nf=Nf
        self.cache_dir=osp.join(root_dir,'cache')
        self.op_cache_dir=osp.join(root_dir,'op_cache')

        # store in memory
        self.verts_list=[]
        self.faces_list=[]
        self.descs_list=[]
        self.elevals_list=[]
        self.elevecs_list=[]  
        self.names_list=[]  
        self.off_files=[]  
        self.combinations=[] 
        self.names_list1=[] 

        if use_cache:
            train_cache=osp.join(self.cache_dir,'train.pt')
            test_cache=osp.join(self.cache_dir,'test.pt')
          
            load_cache=train_cache if self.train else test_cache
            print('using dataset cache path:' + str(load_cache))

            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                # to-do load data
                self.verts_list, self.faces_list, self.descs_list, \
                self.massvec_list, self.evals_list, self.evecs_list,\
                     self.gs_list,self.gradX_list, self.gradY_list, self.L_list,self.elevals_list,self.elevecs_list ,self.names_list,self.off_files,self.combinations= torch.load(load_cache)

                # self.combinations = list(permutations(range(len(self.evals_list)), 2))
                # with open('/home/mj/RFMNet/data/DT4D/intra.txt', 'w') as file:
                #     for ind in range(len(self.combinations)):
                #         ind=self.combinations[ind]
                #         ind1=ind[0]
                #         ind2=ind[1]
                #         file.write(f"{self.names_list[ind1]} {self.names_list[ind2]}"+"\n")
                return
            print("  --> dataset not in cache, repopulating")
        
        self.files='files_train.txt' if self.train else 'files_test.txt'
        self.files=osp.join(root_dir,self.files)
        
        #read files names
        with open(self.files, 'r') as f:
            lines=f.readlines()
            for line in lines:
                line=line.strip()
                if line.split("/")[0] not in ['pumpkinhulk']: 
                    self.off_files += [os.path.join(self.root_dir, 'off', f'{line}.off')]
                    self.names_list1 += [line.split("/")[1]]
        
        # read file and process
        for name in self.names_list1:
            mesh_path=osp.join(root_dir,'shapes',name+'.off')
            verts,faces=pp3d.read_mesh(mesh_path)
            
            #scale the total-area to 1
            sqrt_total_area=np.sqrt(np.sum(vertex_areas(verts,faces)))
            verts=(verts-np.mean(verts,axis=0))/sqrt_total_area
        
            # convert to torch
            verts=torch.from_numpy(verts).float()
            faces=torch.from_numpy(faces)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.names_list.append(name)

            elevals=sio.loadmat(osp.join("/home/mj/ICCV/data/DT4D/elevals",name+'.mat'))['elevals']
            self.elevals_list.append(torch.from_numpy(elevals))
            elevecs=sio.loadmat(osp.join("/home/mj/ICCV/data/DT4D/elevecs_zj",name+'.mat'))['elevecs_zj']
            self.elevecs_list.append(torch.from_numpy(elevecs))
        
        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, \
            self.evals_list, self.evecs_list, self.gradX_list,\
                 self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, \
                     k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)


        # compute meyer filters and hks descriptors
        self.gs_list=[]
        # self.elgs_list=[]
        for s in range(len(self.evals_list)):
            evals=self.evals_list[s]
            self.gs_list.append(Meyer(evals[-1],Nf=self.Nf)(evals))
            evecs=self.evecs_list[s]
            self.descs_list.append(diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16))


        self.inter_cats = set()
        files = os.listdir(os.path.join("/home/mj/h/data/DT4D_r/corres/cross_category_corres"))
        for file in files:
            cat1, cat2 = os.path.splitext(file)[0].split('_')
            self.inter_cats.add((cat1, cat2))
        for i in range(len(self.names_list)):
            for j in range(len(self.names_list)):
                # same category
                cat1, cat2 = self.off_files[i].split('/')[-2], self.off_files[j].split('/')[-2]
                if cat1 != cat2:
                    if (cat1, cat2) in self.inter_cats:
                        self.combinations.append((i, j))           
        
        # save to cache
        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.descs_list, \
                self.massvec_list, self.evals_list, self.evecs_list,\
                     self.gs_list,self.gradX_list, self.gradY_list, self.L_list,self.elevals_list,self.elevecs_list,self.names_list,self.off_files,self.combinations), load_cache)

                        
        # self.combinations = list(permutations(range(len(self.evals_list)), 2))

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, index):
        # get pair data 
        #   descs: [N,P]
        #   massvec: [N,]
        #   evals: [K,]
        #   evecs: [N,K]
        #   gs: [Nf,K,]
        idx1, idx2 = self.combinations[index]
        descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x=self.get_data(idx1)
        descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y=self.get_data(idx2)


        return descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y

    def get_data(self, idx):
        return self.descs_list[idx],self.massvec_list[idx],\
            self.evals_list[idx],self.evecs_list[idx],self.gs_list[idx],\
                self.gradX_list[idx],self.gradY_list[idx],self.elevals_list[idx],self.elevecs_list[idx],self.names_list[idx]
    



                








