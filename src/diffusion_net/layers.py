import sys
import os
import random
# from torch_geometric.nn import knn
import scipy
import scipy.sparse.linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
from .utils import toNP
from .geometry import to_basis, from_basis
import math
import trimesh
from pyFM.mesh import TriMesh
from torch_geometric.data import Batch
from .data import DiffusionData
from .transforms import DiffusionOperatorsTransform
from .prism_decoder import PrismDecoder
from torch_geometric.utils import unbatch
from pyg_lib.ops import grouped_matmul
import roma
from trimesh.graph import face_adjacency

# from prism_decoder import PrismDecoder
class MiniMLPSNK(nn.Sequential):
    """
    A simple MLP with configurable hidden layer sizes.
    """

    def __init__(self, layer_sizes, dropout=0.5, use_bn=True, use_layernorm=False, activation=nn.ReLU, name="mlp"):
        super().__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2) == len(layer_sizes)

            if dropout > 0. and i > 0:
                self.add_module(f"{name}_layer_dropout_{i:03d}", nn.Dropout(dropout))

            # Affine map
            self.add_module(f"{name}_layer_{i:03d}", nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            if use_bn:
                self.add_module(f"{name}_layer_bn_{i:03d}", nn.BatchNorm1d(layer_sizes[i + 1]))

            if use_layernorm:
                self.add_module(f"{name}_layer_layernorm_{i:03d}", nn.LayerNorm(layer_sizes[i + 1]))

            # non-linearity (but not on the last layer)
            if not is_last:
                self.add_module(f"{name}_act_{i:03d}", activation())


class LearnedTimeDiffusionSNK(nn.Module):
    def __init__(self, C_inout, method="spectral", init_time=None, init_std=2.0):
        super().__init__()
        self.C_inout = C_inout
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)

        if init_time is None:
            nn.init.constant_(self.diffusion_time, 0.0)
        else:
            assert isinstance(init_time, (int, float)), "`init_time` must be a scalar"
            nn.init.normal_(self.diffusion_time, mean=init_time, std=init_std)

    def forward(self, x, mass, evals, evecs, batch):
        bs = int(batch.max() + 1)
        neig = evals.size(0) // bs

        # todo do we need to do clipping here? do we need to remove do torch.no_grad? is abs ok (Nick didn't use it initially)?
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        # Transform to spectral
        x_spec = to_basisSNK(x, evecs, mass, batch)

        # Diffuse
        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * self.diffusion_time.unsqueeze(0))
        x_diffuse_spec = diffusion_coefs * x_spec

        # Transform back to per-vertex
        evecs = unbatch(evecs, batch, dim=0)
        x_diffuse_spec = x_diffuse_spec.split(neig)
        x_diffuse = torch.cat(grouped_matmul(evecs, x_diffuse_spec), dim=0)
        return x_diffuse


class SpatialGradientFeaturesSNK(nn.Module):
    """
    Compute dot-products between input vectors. Uses a learned complex-linear layer to keep dimension down.

    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots
    """

    def __init__(self, C_inout, with_gradient_rotations=True):
        super().__init__()

        self.C_inout = C_inout
        self.with_gradient_rotations = with_gradient_rotations

        if self.with_gradient_rotations:
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

        # self.norm = nn.InstanceNorm1d(C_inout)

    def forward(self, vectors):

        vectorsA = vectors  # (V,C)

        if self.with_gradient_rotations:
            vectorsBreal = self.A_re(vectors[..., 0]) - self.A_im(vectors[..., 1])
            vectorsBimag = self.A_re(vectors[..., 1]) + self.A_im(vectors[..., 0])
        else:
            vectorsBreal = self.A(vectors[..., 0])
            vectorsBimag = self.A(vectors[..., 1])

        dots = vectorsA[..., 0] * vectorsBreal + vectorsA[..., 1] * vectorsBimag

        return torch.tanh(dots)


def to_basisSNK(values, basis, massvec, batch):
    """
    Transform data in to an orthonormal basis (where orthonormal is wrt to massvec)
    Inputs:
      - values: (V, D)
      - basis: (V, K)
      - massvec: sparse (V, V)
    Outputs:
      - (B,K,D) transformed values
    """
    basis_T = (massvec.t() @ basis).T  # (K,V) @ (V,V) = (K,V)
    basis_T = unbatch(basis_T, batch, dim=1)  # [K x V1, K x V2, ...]
    values = unbatch(values, batch, dim=0)  # [V1 x D, V2 x D, ...]
    return torch.cat(grouped_matmul(basis_T, values), dim=0)

class DiffusionNetBlockSNK(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_width, mlp_hidden_dims=None, n_layers=2, dropout=0.5, use_bn=True, use_layernorm=False,
                 init_time=2.0, init_std=2.0,
                 diffusion_method="spectral", with_gradient_features=True, with_gradient_rotations=True):
        super().__init__()

        # Specified dimensions
        self.C_width = C_width
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C_width] * n_layers
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusionSNK(self.C_width, method=diffusion_method, init_time=init_time,
                                              init_std=init_std)

        self.MLP_C = 2 * self.C_width

        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeaturesSNK(
                self.C_width, with_gradient_rotations=self.with_gradient_rotations
            )
            self.MLP_C += self.C_width

        # MLPs
        self.mlp = MiniMLPSNK([self.MLP_C] + list(self.mlp_hidden_dims) + [self.C_width],
                           dropout=self.dropout,
                           use_bn=use_bn,
                           use_layernorm=use_layernorm,
                           )

        # todo: is this needed?
        # self.bn = nn.BatchNorm1d(C_width)

    def forward(self, surfaces):
        x_in = surfaces.x

        # Diffusion block
        x_diffuse = self.diffusion(x_in, surfaces.mass, surfaces.evals, surfaces.evecs, surfaces.batch)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_gradX = surfaces.gradX @ x_diffuse
            x_gradY = surfaces.gradY @ x_diffuse
            x_grad = torch.stack((x_gradX, x_gradY), dim=-1)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad)

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        # Apply the mlp
        x0_out = self.mlp(feature_combined)

        # Skip connection
        x0_out = x0_out + x_in

        # # apply batch norm # todo: is this needed?
        # x0_out_batch = self.bn(x0_out_batch)

        # update the features
        surfaces.x = x0_out
        return surfaces


class DiffusionNetSNK(nn.Module):
    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, mlp_hidden_dims=None, dropout=0.5,
                 with_gradient_features=True, with_gradient_rotations=True, use_bn=True, use_layernorm=False,
                 init_time=2.0, init_std=2.0):

        super().__init__()

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation

        # MLP options
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # # Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)

        # DiffusionNet blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = DiffusionNetBlockSNK(
                C_width=C_width,
                mlp_hidden_dims=mlp_hidden_dims,
                dropout=dropout,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
                use_bn=use_bn,
                use_layernorm=use_layernorm,
                init_time=init_time,
                init_std=init_std
            )

            self.blocks.append(block)
            self.add_module("block_" + str(i_block), self.blocks[-1])

    def forward(self, surface):
        surface.x = self.first_lin(surface.x)

        # Apply each of the blocks
        for block in self.blocks:
            surface = block(surface)

        # Apply the last linear layer
        surface.x = self.last_lin(surface.x)

        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            surface.x = self.last_activation(surface.x)

        return surface




def calculate_geodesic_error(dist_x, corr_x, corr_y, p2p, return_mean=True):
    ind21 = np.stack([corr_x, p2p[corr_y]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = np.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err
class Meyer(object):
    def __init__(self, lmax, Nf=6, scales=None):

        self.Nf=Nf

        if scales is None:
            scales = (4./(3 * lmax)).cpu() * np.power(2., np.arange(Nf-2, -1, -1))

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

class LearnedTimeDiffusion(nn.Module):
    """
    Applies diffusion with learned per-channel t.

    In the spectral domain this becomes 
        f_out = e ^ (lambda_i t) f_in

    Inputs:
      - values: (V,C) in the spectral domain
      - L: (V,V) sparse laplacian
      - evals: (K) eigenvalues
      - mass: (V) mass matrix diagonal

      (note: L/evals may be omitted as None depending on method)
    Outputs:
      - (V,C) diffused values 
    """

    def __init__(self, C_inout, method='spectral'):
        super(LearnedTimeDiffusion, self).__init__()
        self.C_inout = C_inout
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.method = method # one of ['spectral', 'implicit_dense']

        nn.init.constant_(self.diffusion_time, 0.0)
        

    def forward(self, x, L, mass, evals, evecs):

        # project times to the positive halfspace
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        if self.method == 'spectral':

            # Transform to spectral
            x_spec = to_basis(x, evecs, mass)

            # Diffuse
            time = self.diffusion_time
            diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
            x_diffuse_spec = diffusion_coefs * x_spec

            # Transform back to per-vertex 
            x_diffuse = from_basis(x_diffuse_spec, evecs)
            
        elif self.method == 'implicit_dense':
            V = x.shape[-2]

            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            mat_dense = L.to_dense().unsqueeze(1).expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)
            
            # Solve the system
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method")


        return x_diffuse


class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors. Uses a learned complex-linear layer to keep dimension down.
    
    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots 
    """

    def __init__(self, C_inout, with_gradient_rotations=True):
        super(SpatialGradientFeatures, self).__init__()

        self.C_inout = C_inout
        self.with_gradient_rotations = with_gradient_rotations

        if(self.with_gradient_rotations):
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

        # self.norm = nn.InstanceNorm1d(C_inout)

    def forward(self, vectors):

        vectorsA = vectors # (V,C)

        if self.with_gradient_rotations:
            vectorsBreal = self.A_re(vectors[...,0]) - self.A_im(vectors[...,1])
            vectorsBimag = self.A_re(vectors[...,1]) + self.A_im(vectors[...,0])
        else:
            vectorsBreal = self.A(vectors[...,0])
            vectorsBimag = self.A(vectors[...,1])

        dots = vectorsA[...,0] * vectorsBreal + vectorsA[...,1] * vectorsBimag

        return torch.tanh(dots)


class MiniMLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    activation()
                )


class DiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_width, mlp_hidden_dims,
                 dropout=True, 
                 diffusion_method='spectral',
                 with_gradient_features=True, 
                 with_gradient_rotations=True):
        super(DiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method)
        
        self.MLP_C = 2*self.C_width
      
        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(self.C_width, with_gradient_rotations=self.with_gradient_rotations)
            self.MLP_C += self.C_width
        
        # MLPs
        self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)


    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):

        # Manage dimensions
        B = x_in.shape[0] # batch dimension
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width))
        
        # Diffusion block 
        x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_grads = [] # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching
            for b in range(B):
                # gradient after diffusion
                x_gradX = torch.mm(gradX[b,...], x_diffuse[b,...])
                x_gradY = torch.mm(gradY[b,...], x_diffuse[b,...])

                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
            x_grad = torch.stack(x_grads, dim=0)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad) 

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        
        # Apply the mlp
        x0_out = self.mlp(feature_combined)

        # Skip connection
        x0_out = x0_out + x_in

        return x0_out


class DiffusionNet(nn.Module):

    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, outputs_at='vertices', mlp_hidden_dims=None, dropout=True, 
                       with_gradient_features=True, with_gradient_rotations=True, diffusion_method='spectral'):   
        """
        Construct a DiffusionNet.

        Parameters:
            C_in (int):                     input dimension 
            C_out (int):                    output dimension 
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient. Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(DiffusionNet, self).__init__()

        ## Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ['vertices', 'edges', 'faces', 'global_mean']: raise ValueError("invalid setting for outputs_at")

        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        
        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations
        
        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
       
        # DiffusionNet blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = DiffusionNetBlock(C_width = C_width,
                                      mlp_hidden_dims = mlp_hidden_dims,
                                      dropout = dropout,
                                      diffusion_method = diffusion_method,
                                      with_gradient_features = with_gradient_features, 
                                      with_gradient_rotations = with_gradient_rotations)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    
    def forward(self, x_in, mass, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None, faces=None):
        """
        A forward pass on the DiffusionNet.

        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].

        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet, not all are strictly necessary.

        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """


        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in: 
            raise ValueError("DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(self.C_in,x_in.shape[-1]))
        N = x_in.shape[-2]
        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L != None: L = L.unsqueeze(0)
            if evals != None: evals = evals.unsqueeze(0)
            if evecs != None: evecs = evecs.unsqueeze(0)
            if gradX != None: gradX = gradX.unsqueeze(0)
            if gradY != None: gradY = gradY.unsqueeze(0)
            if edges != None: edges = edges.unsqueeze(0)
            if faces != None: faces = faces.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        
        else: raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")
        
        # Apply the first linear layer
        x = self.first_lin(x_in)
      
        # Apply each of the blocks
        for b in self.blocks:
            x = b(x, mass, L, evals, evecs, gradX, gradY)
        
        # Apply the last linear layer
        x = self.last_lin(x)

        # Remap output to faces/edges if requested
        if self.outputs_at == 'vertices': 
            x_out = x
        
        elif self.outputs_at == 'edges': 
            # Remap to edges
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)
        
        elif self.outputs_at == 'faces': 
            # Remap to faces
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)
        
        elif self.outputs_at == 'global_mean': 
            # Produce a single global mean ouput.
            # Using a weighted mean according to the point mass/area is discretization-invariant. 
            # (A naive mean is not discretization-invariant; it could be affected by sampling a region more densely)
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out





from .utils import dist_mat, nn_search

class RFMNet(torch.nn.Module):
    def __init__(self,C_in, C_out, is_mwp=True):
        
        super().__init__()
        self.is_mwp=is_mwp
        self.feat_extrac=DiffusionNet(C_in=C_in, C_out=C_out)
        self.criterion=torch.nn.MSELoss()


    def forward(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y):

        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)
# #双MWP总


        elevals_x=elevals_x[0][:220]
        elevals_y=elevals_y[0][:220]  
        elevecs_x=elevecs_x[:,:220]
        elevecs_y=elevecs_y[:,:220]  

        elevecs_x=elevecs_x.to(dtype=torch.float32)
        elevecs_y=elevecs_y.to(dtype=torch.float32)  

        BT_A_B = elevecs_x.T @ torch.diag(massvec_x) @ elevecs_x
        BT_A_B_inv_sqrt = torch.linalg.pinv(torch.linalg.cholesky(BT_A_B))
        elevecs_x = elevecs_x @ BT_A_B_inv_sqrt

        BT_A_B = elevecs_y.T @ torch.diag(massvec_y) @ elevecs_y
        BT_A_B_inv_sqrt = torch.linalg.pinv(torch.linalg.cholesky(BT_A_B))
        elevecs_y = elevecs_y @ BT_A_B_inv_sqrt

        elgs_x=[]
        positive_evals_x = elevals_x[elevals_x > 0].cpu()
        negative_evals_x = elevals_x[elevals_x < 0].cpu()
        for i in range(len(elevals_x)):
            if elevals_x[i]>0:
                elgs_x.append(Meyer(positive_evals_x[-1],Nf=6)(elevals_x[i].cpu()))
            else:
                elgs_x.append(Meyer(negative_evals_x[-1],Nf=6)(elevals_x[i].cpu())) 
        elgs_x=torch.stack(elgs_x)
        elgs_x=elgs_x.T
        elgs_x=elgs_x.to("cuda:0")

        elgs_y=[]
        positive_evals_y = elevals_y[elevals_y > 0].cpu()
        negative_evals_y = elevals_y[elevals_y < 0].cpu()
        for i in range(len(elevals_y)):
            if elevals_y[i]>0:
                elgs_y.append(Meyer(positive_evals_y[-1],Nf=6)(elevals_y[i].cpu()))
            else:
                elgs_y.append(Meyer(negative_evals_y[-1],Nf=6)(elevals_y[i].cpu())) 
        elgs_y=torch.stack(elgs_y)
        elgs_y=elgs_y.T
        elgs_y=elgs_y.to("cuda:0")
    


        Q_ni = torch.linalg.pinv(elevecs_x)
        M_ni = torch.linalg.pinv(elevecs_y)



        p121=nn_search(feat_y,feat_x)
        # p122=nn_search(feat_x,feat_y)#yuanwu
        p12=self.soft_correspondence(feat_x,feat_y)
        # dims=(feat_x.shape[0],feat_y.shape[0])#测试是否进行反向传播
        # p12=torch.rand(dims)
        # p12=p12.to("cuda")

#MWP
        # p12el=self.soft_correspondence(elevecs_x,elevecs_y@Cel.t())   
        # p12=p12.to(torch.float32)  
        # p12el=p12el.to(torch.float32)
        # clb=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(p12@evecs_y))
        # if self.is_mwp:
        #     Clb=self.MWP(gs_x, gs_y,clb)
        # p122=nn_search(evecs_y@Clb.t(), evecs_x)
        # for _ in range(3):
        #     clb=evecs_x.transpose(-2,-1)@(torch.diag(massvec_x))@evecs_y[p121,:]
        #     if self.is_mwp:
        #         Clb=self.MWP(gs_x, gs_y,clb)
        #     p121=nn_search(evecs_y@Clb.t(), evecs_x)     
#MWP   
        # p121=nn_search(feat_y,feat_x)
        # self.feat_correspondences(feat_x,feat_y)
        # for _ in range(3):            
        #     C21=evecs_x.transpose(-2,-1)@(torch.diag(massvec_x))@evecs_y[p121,:]
        #     # self.C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(self.p@evecs_y))
        #     if self.is_mwp:
        #         C211=self.MWP(gs_x, gs_y,C21)
        #     p121=nn_search(evecs_y@C211.t(), evecs_x) #MWP         

        # loss=self.criterion(evecs_x,self.p@evecs_y@C21.transpose(-2,-1))

        # loss=self.criterion(evecs_x,self.p@evecs_y@fm.transpose(-2,-1))

        # return loss

#弹性MWP
        # p12=p12.to(torch.float64)
        # cel=Q_ni@p12@elevecs_y#结果要差
        # if self.is_mwp:
        #     Cel=self.MWP(elgs_x, elgs_y,cel) 
        # p124=nn_search(elevecs_y@Cel.t(), elevecs_x) 
        # for _ in range(3):
        #     C_fmapel = Q_ni @ elevecs_y[p121, :]#结果要好
        #     if self.is_mwp:
        #         Cel=self.MWP(elgs_x, elgs_y,C_fmapel) 
        #     p121=nn_search(elevecs_y@Cel.t(), elevecs_x)    
#弹性MWP     

#双MWP：
        for it in range(3):
            
            C_fmapel = Q_ni @ elevecs_y[p121, :]#结果要好原
            # C_fmapel = M_ni @ elevecs_x[p122, :]#结果要好
            if self.is_mwp:
                Cel=self.MWP(elgs_x, elgs_y,C_fmapel)
            # Cel=torch.inverse(AM)@Cel.t()@AN
            p126=nn_search(elevecs_y@Cel.t(), elevecs_x) 
            # Cel=Cel@AN_sqrt
            # p126=nn_search(elevecs_y@AN_sqrt, elevecs_x@Cel) 
            clb=evecs_x.transpose(-2,-1)@(torch.diag(massvec_x))@evecs_y[p126,:]
            if self.is_mwp:
                 Clb=self.MWP(gs_x, gs_y,clb)
            # com_1=torch.cat((evecs_y@ Clb.T,elevecs_y @ Cel.T),dim=1)
            # com_2=torch.cat((evecs_x, elevecs_x),dim=1)      
            com_1=torch.cat((evecs_y@ Clb.T,elevecs_y@Cel.T),dim=1)
            com_2=torch.cat((evecs_x, elevecs_x),dim=1)       
            p128 = nn_search(com_1,com_2)
            p121=p128
# #双MWP总
        # p12=p12.double()
        # loss=self.criterion(evecs_x,p12@evecs_y@Clb.transpose(-2,-1))
        loss=self.criterion(evecs_x,p12@evecs_y@Clb.transpose(-2,-1))
        # loss=self.criterion(evecs_x@Clb,p12@evecs_y)


        return loss,feat_y,p121

    
    def sinkhorn(self, d, sigma=0.1, num_sink=10):
        d = d / d.mean()
        log_p = -d / (2*sigma**2)
        
        for it in range(num_sink):
            log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        self.p = torch.exp(log_p)
        log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        self.p_adj = torch.exp(log_p).transpose(0, 1)

    def soft_correspondence(self, emb_x,emb_y, sigma = 1e2):
        emb_x = emb_x / emb_x.norm(dim=1, keepdim=True)
        emb_y = emb_y / emb_y.norm(dim=1, keepdim=True)

        D = torch.matmul(emb_x, emb_y.transpose(0, 1))

        
        self.p = torch.nn.functional.softmax(D * sigma, dim=1)
        self.pdj=torch.nn.functional.softmax(D * sigma, dim=0).transpose(0, 1)
        return self.p

    def feat_correspondences(self, emb_x, emb_y):
        d = dist_mat(emb_x, emb_y, False)
        self.sinkhorn(d)

    def MWP(self, gs_x, gs_y,C):
        # input:
        #   massvec_x/y: [M/N,]
        #   evecs_x/y: [M/N,Kx/Ky]
        #   gs_x/y: [Nf,Kx/Ky]

        # compute MWP functional map
        # C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(self.p@evecs_y))
        C_new=torch.zeros_like(C)

        gs_x=gs_x.unsqueeze(-1) # [Nf,Kx]->[Nf,Kx,1]
        gs_y=gs_y.unsqueeze(-1) # [Nf,Ky]->[Nf,Ky,1]

        # MWP filters
        Nf=gs_x.size(0)
        for s in range(Nf):
            C_new+=gs_x[s]*C*gs_y[s].transpose(-2,-1)
        
        return C_new
    
    def model_test_opt(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y):

        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)

# #双MWP总

        elevals_x=elevals_x[0][:200]
        elevals_y=elevals_y[0][:200]  
        elevecs_x=elevecs_x[:,:200]
        elevecs_y=elevecs_y[:,:200]   

        elevecs_x=elevecs_x.to(dtype=torch.float32)
        elevecs_y=elevecs_y.to(dtype=torch.float32) 

        BT_A_B = elevecs_x.T @ torch.diag(massvec_x) @ elevecs_x
        BT_A_B_inv_sqrt = torch.linalg.pinv(torch.linalg.cholesky(BT_A_B))
        elevecs_x = elevecs_x @ BT_A_B_inv_sqrt

        BT_A_B = elevecs_y.T @ torch.diag(massvec_y) @ elevecs_y
        BT_A_B_inv_sqrt = torch.linalg.pinv(torch.linalg.cholesky(BT_A_B))
        elevecs_y = elevecs_y @ BT_A_B_inv_sqrt

        elgs_x=[]
        positive_evals_x = elevals_x[elevals_x > 0].cpu()
        negative_evals_x = elevals_x[elevals_x < 0].cpu()
        for i in range(len(elevals_x)):
            if elevals_x[i]>0:
                elgs_x.append(Meyer(positive_evals_x[-1],Nf=6)(elevals_x[i].cpu()))
            else:
                elgs_x.append(Meyer(negative_evals_x[-1],Nf=6)(elevals_x[i].cpu())) 
        elgs_x=torch.stack(elgs_x)
        elgs_x=elgs_x.T
        elgs_x=elgs_x.to("cuda:0")

        elgs_y=[]
        positive_evals_y = elevals_y[elevals_y > 0].cpu()
        negative_evals_y = elevals_y[elevals_y < 0].cpu()
        for i in range(len(elevals_y)):
            if elevals_y[i]>0:
                elgs_y.append(Meyer(positive_evals_y[-1],Nf=6)(elevals_y[i].cpu()))
            else:
                elgs_y.append(Meyer(negative_evals_y[-1],Nf=6)(elevals_y[i].cpu())) 
        elgs_y=torch.stack(elgs_y)
        elgs_y=elgs_y.T
        elgs_y=elgs_y.to("cuda:0")
        
        Q_ni = torch.linalg.pinv(elevecs_x)
        M_ni = torch.linalg.pinv(elevecs_y)
        
        # Q_ni = elevecs_x.t()@torch.diag(massvec_x)
        # M_ni = elevecs_y.t()@torch.diag(massvec_y)

        # Q_ni = elevecs_x.t()
        # M_ni = elevecs_y.t()
 #直接特征最近邻
        p121=nn_search(feat_y,feat_x)
        # C_fmapel = Q_ni @ elevecs_y[p121, :]#结果要好
        # if self.is_mwp:
        #     Cel=self.MWP(elgs_x, elgs_y,C_fmapel) 
        # p126=nn_search(elevecs_y@Cel.t(), elevecs_x) 
#直接特征最近邻
        # p121=sio.loadmat("/home/mj/ICCV/data/smal/p12/"+name_x+"_"+name_y+".mat")
        # p121=p121["p12"] 
        # p121=torch.tensor(p121)
        # p121=p121[0] 
        # p121=p121-1
        for it in range(3):
            
            C_fmapel = Q_ni @ elevecs_y[p121, :]#结果要好
            if self.is_mwp:
                Cel=self.MWP(elgs_x, elgs_y,C_fmapel) 
            p126=nn_search(elevecs_y@Cel.t(), elevecs_x) 

            clb=evecs_x.transpose(-2,-1)@(torch.diag(massvec_x))@evecs_y[p126,:]
            if self.is_mwp:
                Clb=self.MWP(gs_x, gs_y,clb)

            com_1=torch.cat((evecs_y@ Clb.T,elevecs_y @ Cel.T),dim=1)
            com_2=torch.cat((evecs_x, elevecs_x),dim=1)


            
            p128 = nn_search(com_1,com_2)
            p121=p128
#双MWP总
#         #先转换泛函映射再转成逐点映射
        # evecs_trans_x=(evecs_x*massvec_x.unsqueeze(-1)).t()
        # Pxy=self.soft_correspondence(feat_x,feat_y)
        # Cyx = evecs_trans_x @ (Pxy @ evecs_y)
        # p121 = nn_search(evecs_y@(Cyx.t()), evecs_x)

        # self.feat_correspondences(feat_x,feat_y)#MWP
        # self.C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(self.p@evecs_y))
        
        # self.C=self.MWP(gs_x, gs_y,self.C)
        # T=nn_search(evecs_y@self.C.t(), evecs_x) #MWP       


        # p121=nn_search(feat_y,feat_x)
        # for _ in range(3):            
        #     C21=evecs_x.transpose(-2,-1)@(torch.diag(massvec_x))@evecs_y[p121,:]
        #     if self.is_mwp:
        #         C211=self.MWP(gs_x, gs_y,C21)
        #     p121=nn_search(evecs_y@C211.t(), evecs_x) #MWP  

        # p121=nn_search(feat_y,feat_x)
        # # self.feat_correspondences(feat_x, feat_y)
        # for _ in range(3):            
        #     C21=evecs_x.transpose(-2,-1)@(torch.diag(massvec_x))@evecs_y[p121,:]
        # # self.C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(self.p@evecs_y))
        # # self.MWP(gs_x, gs_y)
        #     if self.is_mwp:
        #         C211=self.MWP(gs_x, gs_y,C21)
        #     p121=nn_search(evecs_y@C211.t(), evecs_x) #WMP
        # T = nn_search(evecs_y @ self.C.t(), evecs_x)
        return p121
    def model_test_nn(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y):
        
        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)

        # get initialial map T0 using nn_search
        T0=nn_search(feat_y,feat_x)
        # axioMWP 
        C,T=self.axioMWP(massvec_x,evecs_x,gs_x,evecs_y,gs_y,T0,num_iter=1)

        return C,T
    
    def model_test_with_axioMWP(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y):
        
        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)

        # get initialial map T0 using nn_search
        T0=nn_search(feat_y,feat_x)
        # axioMWP 
        C,T=self.axioMWP(massvec_x,evecs_x,gs_x,evecs_y,gs_y,T0)

        return C,T


    def axioMWP(self,massvec_x,evecs_x,gs_x,evecs_y,gs_y,T,num_iter=5):
        # input: 
        #   massvec_x: [M,]
        #   evecs_x/y: [M/N,Kx/Ky]
        #   gs_x/y: [Nf,Kx/Ky]
        #   T: [M,]
        gs_x=gs_x.unsqueeze(-1) # [Nf,Kx]->[Nf,Kx,1]
        gs_y=gs_y.unsqueeze(-1) # [Nf,Ky]->[Nf,Ky,1]
        Nf=gs_x.size(0)
        
        for it in range(num_iter):
            C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(evecs_y[T,:]))
            C_new=torch.zeros_like(C)
            
            for s in range(Nf):
                C_new+=gs_x[s]*C*gs_y[s].transpose(-2,-1)
            
            T=nn_search(evecs_y@C.t(),evecs_x)
        
        return C_new, T


class SSWFMNet(nn.Module):
    def __init__(self, C_in,  C_out,resolvant_gamma=0.5):
        super().__init__()
        # self.n_fmap=n_fmap
        self.feat_extrac=DiffusionNet(C_in=C_in, C_out=C_out)
        # self.frob_loss = FrobeniusLoss()
        # self.lambda_param= torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.lambda_param.data.fill_(1e-3)
        self.lambda_param=1e-13
        # self.lambda_param=torch.tensor([1e-3],requires_grad=True)
        # self.lambda_param=self.lambda_param.to("cuda:1")
        self.resolvant_gamma = resolvant_gamma
        self.criterion=torch.nn.MSELoss()
        
        

    def forward(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y):
        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        # feat_x_1=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)        
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)
        feat_x = feat_x / feat_x.norm(dim=1, keepdim=True)
        feat_y = feat_y / feat_y.norm(dim=1, keepdim=True)
        # feat_x=torch.pow(feat_x,2)
        # feat_y=torch.pow(feat_y,2)

        evecs_x=evecs_x[:,:220]
        evecs_y=evecs_y[:,:220]
        gs_x=gs_x[:,:220]
        gs_y=gs_y[:,:220]
        #使用多尺度小波作为正则项优化C
        evecs_trans_x=(evecs_x*massvec_x.unsqueeze(-1)).t()
   
        evecs_trans_y=(evecs_y*massvec_y.unsqueeze(-1)).t()
       
        C21=self.waveletCReg(feat_y, feat_x, evecs_trans_y, evecs_trans_x, gs_y, gs_x)
    

        # #利用sinkhorn算法 更准确估计逐点映射Pxy
        Pxy=self.feat_correspondences(feat_x,feat_y)

 
        loss=self.criterion(evecs_x, Pxy@evecs_y@C21.transpose(-2,-1))
  

        
        return loss
    
    # def get_lambda_param(self):
    #     return self.lambda_param
    # model = SS
    # a =  model.get
    def model_test_opt(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y):

        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)
        feat_x = feat_x / feat_x.norm(dim=1, keepdim=True)
        feat_y = feat_y / feat_y.norm(dim=1, keepdim=True)
       
        evecs_x=evecs_x[:,:100]
        evecs_y=evecs_y[:,:100]
        # massvec_x=massvec_x.diagonal()
        # massvec_y=massvec_y.diagonal()
        #使用多尺度小波作为正则项优化C
        evecs_trans_x=(evecs_x*massvec_x.unsqueeze(-1)).t()
        evecs_trans_y=(evecs_y*massvec_y.unsqueeze(-1)).t()
        
        Pxy=self.feat_correspondences(feat_x,feat_y)#iso
  
        Cyx = evecs_trans_x @ (Pxy @ evecs_y)#iso
      
        p122 = nn_search(evecs_y@(Cyx.t()), evecs_x)#iso   yx
   
        return p122
    
    def MWP(self, gs_x, gs_y):
        # input:
        #   massvec_x/y: [M/N,]
        #   evecs_x/y: [M/N,Kx/Ky]
        #   gs_x/y: [Nf,Kx/Ky]

        # compute MWP functional map
        # C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(self.p@evecs_y))
        C_new=torch.zeros_like(self.C)

        gs_x=gs_x.unsqueeze(-1) # [Nf,Kx]->[Nf,Kx,1]
        gs_y=gs_y.unsqueeze(-1) # [Nf,Ky]->[Nf,Ky,1]

        # MWP filters
        Nf=gs_x.size(0)
        for s in range(Nf):
            C_new+=gs_x[s]*self.C*gs_y[s].transpose(-2,-1)
        
        self.C=C_new
        

    def sinkhorn(self, d, sigma=0.1, num_sink=10):
        d = d / d.mean()
        log_p = -d / (2*sigma**2)
        
        for it in range(num_sink):
            log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        # self.p = torch.exp(log_p)
        p = torch.exp(log_p)
        return p
        # log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        # self.p_adj = torch.exp(log_p).transpose(0, 1)
    
    def feat_correspondences(self, emb_x, emb_y):
        d = dist_mat(emb_x, emb_y, False)
        return self.sinkhorn(d)
        
    
    def soft_correspondence(self, emb_x,emb_y, sigma = 1e2):
        emb_x = emb_x / emb_x.norm(dim=1, keepdim=True)
        emb_y = emb_y / emb_y.norm(dim=1, keepdim=True)

        D = torch.matmul(emb_x, emb_y.transpose(0, 1))

        p=torch.nn.functional.softmax(D * sigma, dim=1)
        return p
        # self.p = torch.nn.functional.softmax(D * sigma, dim=1)
        # self.pdj=torch.nn.functional.softmax(D * sigma, dim=0).transpose(0, 1)
        
    # 小波正则项
    
    def waveletCReg(self, feat_x, feat_y, evecs_trans_x, evecs_trans_y, gs_x, gs_y):
        # feat_x/y; [nx/ny,p]
        # gs_x/y; [s,kx/ky]
        # evecs_trans_x/y: [kx/ky,nx/ny]sinkhorn
        # output: Cxy

        #计算傅里叶(谱)系数
        A = torch.matmul(evecs_trans_x, feat_x)
        B = torch.matmul(evecs_trans_y, feat_y)
        scaling_factor = max(torch.max(gs_x), torch.max(gs_y))
        gs_x, gs_y = gs_x / scaling_factor, gs_y / scaling_factor
        #构造W矩阵
        Ds=0
        # w=[self.w_1,self.w_2,self.w_3,self.w_4,self.w_5,self.w_6]
        Nf = gs_x.size(0)
        for s in range(Nf):
            # D=(gs_y[s].unsqueeze(1)-gs_x[s].unsqueeze(0))**2 #利用广播机制

            D=(gs_x[s].unsqueeze(0)-gs_y[s].unsqueeze(1))**2 
            Ds+=D
            
        # Ds=torch.cat(Ds,dim=0)
        # Ds=torch.sum(Ds,dim=0)

        #计算C
        A_A_T=torch.matmul(A,A.t())
        A_B_T=torch.matmul(A,B.t())#A,B顺序
        
        C=[]
        for i in range(gs_y.size(1)):
            D_i=torch.diag(Ds[i])
            # C_i= torch.matmul(torch.inverse(A_A_T +lambda_param * D_i), A_B_T[:, i])
            C_i=torch.linalg.lstsq(A_A_T+self.lambda_param*D_i,A_B_T[:,i]).solution
            C.append(C_i.unsqueeze(0))

        C=torch.cat(C,dim=0)

        return C


class PrismDecoder(torch.nn.Module):
    def __init__(self, dim_in, dim_out, n_width=256, n_block=4):
        # , pairwise_dot=True, dropout=False, dot_linear_complex=True, neig=128):
        super().__init__()

        self.diffusion_net = DiffusionNetSNK(C_in=dim_in, C_out=dim_out, C_width=n_width, N_block=n_block)
                                        #   , pairwise_dot=pairwise_dot, dropout=dropout,
                                        #   dot_linear_complex=dot_linear_complex, neig=neig)

        # self.mlp_refine = MLP(dim_out)
        
        self.mlp_refine = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
        )

    def forward(self, batch):
        # original prism
        verts = batch.pos.reshape(-1, 3)
        faces = batch.face.t()
        prism_base = verts[faces]  # (n_faces, 3, 3)
        bs, _, _ = batch.pos.shape #

        # forward through diffusion net
        batch = self.diffusion_net(batch)  # (bs, n_verts, dim)

        # features per face
        x = batch.x.unsqueeze(0)
        x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
        faces_gather = faces.unsqueeze(0).unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
        xf = torch.gather(x_gather, 1, faces_gather)
        features = torch.mean(xf, dim=-1)  # (bs, n_faces, dim)

        # refine features with mlp
        features = self.mlp_refine(features)  # (bs, n_faces, 12)

        # get the translation and rotation
        rotations = features[:, :, :9].reshape(-1, 3, 3)
        rotations = roma.special_procrustes(rotations)  # (n_faces, 3, 3)
        translations = features[:, :, 9:].reshape(-1, 3)  # (n_faces, 3)

        # transform the prism
        transformed_prism = (prism_base @ rotations) + translations[:, None]

        # prism to vertices
        features = self.prism_to_vertices(transformed_prism, faces, verts)

        batch.features = features.reshape(bs, -1, 3)
        batch.transformed_prism = transformed_prism
        batch.rotations = rotations
        return batch

    def prism_to_vertices(self, prism, faces, verts):
        # initialize the transformed features tensor
        N = verts.shape[0]
        d = prism.shape[-1]
        device = prism.device
        features = torch.zeros((N, d), device=device)

        # scatter the features in K onto L using the indices in F
        features.scatter_add_(0, faces[:, :, None].repeat(1, 1, d).reshape(-1, d), prism.reshape(-1, d))

        # divide each row in the transformed features tensor by the number of faces that the corresponding vertex appears in
        num_faces_per_vertex = torch.zeros(N, dtype=torch.float32, device=device)
        num_faces_per_vertex.index_add_(0, faces.reshape(-1), torch.ones(faces.shape[0] * 3, device=device))
        features /= num_faces_per_vertex.unsqueeze(1).clamp(min=1)

        return features
        
class PrismRegularizationLoss(nn.Module):
    """
    Calculate the loss based on the PriMo energy, as described in the paper:
    PriMo: Coupled Prisms for Intuitive Surface Modeling
    """
    def __init__(self, primo_h=1):
        super().__init__()
        self.h = primo_h

        # compute coefficient for the energy
        indices = torch.tensor([(i, j) for i in range(2) for j in range(2)])
        indices_A = indices.repeat_interleave(4, dim=0)
        indices_B = indices.repeat(4, 1)
        self.coeff = (torch.ones(1) * 2).pow(((indices_A - indices_B).abs() * -1).sum(dim=1))[None, :]

    def compute_normals(self, transformed_prism):
        normals = []
        for face in transformed_prism:
            # Calculate vectors for the edges
            edge1 = face[0] - face[1]
            edge2 = face[0] - face[2]
            # Compute the cross product of the edges to get the normal vectors
            normal = torch.cross(edge1, edge2)
            # Normalize the normals
            normal = normal / normal.norm(dim=0, keepdim=True)
            normals.append(normal)
        return normals
    
    def forward(self, transformed_prism, rotations, verts, faces):
        # transformed_prism is (n_faces, 3, 3)
        # verts and faces are from the template (shape 2)
        # * for now assumes there is only one batch
        # todo add batch support

        # computed normals from the transformed prism
        normals = self.compute_normals(transformed_prism)

        bs = verts.shape[0]
        verts = verts.reshape(-1, 3)
        normals = torch.stack(normals)
        faces = faces.t()

        # get the area of each face
        face_areas = self.get_face_areas(verts, faces)  # (n_faces,)

        # get list of edges and the faces that share each edge
        face_ids, edges = face_adjacency(faces.cpu().numpy(), return_edges=True)  # (n_edges, 2), (n_edges, 2)
        face_ids, edges = torch.from_numpy(face_ids).to(verts.device), torch.from_numpy(edges).to(verts.device)

        # normals and rotations of the faces that share each edge
        normals1, normals2 = normals[edges[:, 0]], normals[edges[:, 1]]  # (n_edges, 3), normals are per vertex
        rotations1, rotations2 = rotations[face_ids[:, 0]], rotations[face_ids[:, 1]]  # (n_edges, 3, 3), rotations are per face

        # computed normals from the transformed prism
        # normals = self.compute_normals(transformed_prism)

        # compute the loss
        face_id1, face_id2 = face_ids[:, 0], face_ids[:, 1]  # (n_edges,)
        faces_to_verts = self.get_verts_id_face(faces, edges, face_ids)  # (n_edges, 4)
        verts1_p1, verts2_p1 = transformed_prism[face_id1, faces_to_verts[:, 0]], transformed_prism[face_id1, faces_to_verts[:, 1]]  # (n_edges, 3)
        verts1_p2, verts2_p2 = transformed_prism[face_id2, faces_to_verts[:, 2]], transformed_prism[face_id2, faces_to_verts[:, 3]]  # (n_edges, 3)

        # get the normals per vertex
        # normals1, normals2 = normals[face_id1], normals[face_id2]  # (n_edges, 3)  # normals per face (NOT USED)
        prism1_n1, prism1_n2 = (normals1[:, None] @ rotations1).squeeze(1), (normals2[:, None] @ rotations1).squeeze(1)  # todo check if this is correct
        prism2_n1, prism2_n2 = (normals1[:, None] @ rotations2).squeeze(1), (normals2[:, None] @ rotations2).squeeze(1)

        # get the coordinates of the face of the prism
        # prism1 (1 -> 2)
        f_p1_00, f_p1_01 = verts1_p1 + prism1_n1 * self.h, verts2_p1 + prism1_n2 * self.h  # (n_edges, 3)
        f_p1_10, f_p1_11 = verts1_p1 - prism1_n1 * self.h, verts2_p1 - prism1_n2 * self.h  # (n_edges, 3)
        # prism2 (2 -> 1)
        f_p2_00, f_p2_01 = verts1_p2 + prism2_n1 * self.h, verts2_p2 + prism2_n2 * self.h  # (n_edges, 3)
        f_p2_10, f_p2_11 = verts1_p2 - prism2_n1 * self.h, verts2_p2 - prism2_n2 * self.h  # (n_edges, 3)

        # compute the energy
        A, B = torch.stack((f_p1_00, f_p1_01, f_p1_10, f_p1_11), dim=1), torch.stack((f_p2_00, f_p2_01, f_p2_10, f_p2_11), dim=1)  # (n_edges, 4, 3)
        energy = self.compute_energy(A - B, A - B)  # (n_edges,)

        # compute weight
        area1, area2 = face_areas[face_id1], face_areas[face_id2]  # (n_edges,)
        weight = torch.norm(verts[edges[:, 0]] - verts[edges[:, 1]], dim=1).square() / (area1 + area2)  # (n_edges,)
        # weight = torch.ones_like(weight).to(weight.device)  # todo remove
        energy = energy * weight  # (n_edges,)

        loss = energy.sum() / bs  # todo when batch enabled, need to divide by batch size
        return loss

    def compute_energy(self, A, B):
        """
        Computes the formula sum_{i,j,k,l=0}^{1} a_{ij}b_{kl} 2^{-|i - k| - |j - l|}.
        Assumes that A and B are tensors of size bs x 4 x 3, where bs is the batch size.
        """
        self.coeff = self.coeff.to(A.device)

        A_repeated = A.repeat_interleave(4, dim=1)
        B_repeated = B.repeat(1, 4, 1)

        energy = (A_repeated * B_repeated).sum(dim=-1)
        energy = (energy * self.coeff).sum(dim=1)
        energy = energy / 9

        return energy

    def get_face_areas(self, verts, faces):
        # get the area of each face
        v1, v2, v3 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        area = 0.5 * torch.cross(v2 - v1, v3 - v1).norm(dim=1)

        return area

    def get_verts_id_face(self, F, E, Q):
        e = E.shape[0]
        Z = torch.zeros((e, 4), dtype=torch.long)

        v1 = F[:, 0][Q[:, 0]]
        v2 = F[:, 1][Q[:, 0]]
        v3 = F[:, 2][Q[:, 0]]
        v4 = F[:, 0][Q[:, 1]]
        v5 = F[:, 1][Q[:, 1]]
        v6 = F[:, 2][Q[:, 1]]

        idx1 = torch.where(v1 == E[:, 0], 0, torch.where(v2 == E[:, 0], 1, torch.where(v3 == E[:, 0], 2, -1)))
        idx2 = torch.where(v1 == E[:, 1], 0, torch.where(v2 == E[:, 1], 1, torch.where(v3 == E[:, 1], 2, -1)))
        idx3 = torch.where(v4 == E[:, 0], 0, torch.where(v5 == E[:, 0], 1, torch.where(v6 == E[:, 0], 2, -1)))
        idx4 = torch.where(v4 == E[:, 1], 0, torch.where(v5 == E[:, 1], 1, torch.where(v6 == E[:, 1], 2, -1)))

        Z[:, 0:2] = torch.stack((idx1, idx2), dim=1)
        Z[:, 2:4] = torch.stack((idx3, idx4), dim=1)
        Z = Z.to(F.device)

        return Z

class RFMNet11(torch.nn.Module):
    def __init__(self,C_in, C_out, is_mwp=True):
        
        super().__init__()
        self.is_mwp=is_mwp
        self.feat_extrac=DiffusionNet(C_in=C_in, C_out=C_out)
        self.criterion=torch.nn.MSELoss()
        # self.decoder=PrismDecoder()
    

    def forward(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y):

        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)

        p12=self.feat_correspondences(feat_x,feat_y)
        # p121=nn_search(feat_y,feat_x)
        # self.C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(p12@evecs_y))
        # p121=nn_search(evecs_y@self.C.t(), evecs_x) #MWP  
        # for _ in range(3):            
        #     self.C=evecs_x.transpose(-2,-1)@(torch.diag(massvec_x))@evecs_y[p121,:]
        #     # self.C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(self.p@evecs_y))
        #     if self.is_mwp:
        #         self.MWP(gs_x, gs_y)
        #     p121=nn_search(evecs_y@self.C.t(), evecs_x) #MWP   
        # self.C=evecs_x.transpose(-2,-1)@(torch.diag(massvec_x))@evecs_y[p121,:]
        self.C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(p12@evecs_y))
        if self.is_mwp:
            self.MWP(gs_x, gs_y)
        p121=nn_search(evecs_y@self.C.t(), evecs_x)


        loss1=self.criterion(evecs_x,p12@evecs_y@self.C.transpose(-2,-1))


        # loss=0*loss1+0.1*loss_E+loss2

        return loss1,feat_y,p121

    
    def sinkhorn(self, d, sigma=0.1, num_sink=10):
        d = d / d.mean()
        log_p = -d / (2*sigma**2)
        
        for it in range(num_sink):
            log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        self.p = torch.exp(log_p)
        # log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        # self.p_adj = torch.exp(log_p).transpose(0, 1)
        return self.p
    
    def feat_correspondences(self, emb_x, emb_y):
        d = dist_mat(emb_x, emb_y, False)
        p12=self.sinkhorn(d)
        return p12

    def MWP(self, gs_x, gs_y):
        # input:
        #   massvec_x/y: [M/N,]
        #   evecs_x/y: [M/N,Kx/Ky]
        #   gs_x/y: [Nf,Kx/Ky]

        # compute MWP functional map
        # C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(self.p@evecs_y))
        C_new=torch.zeros_like(self.C)

        gs_x=gs_x.unsqueeze(-1) # [Nf,Kx]->[Nf,Kx,1]
        gs_y=gs_y.unsqueeze(-1) # [Nf,Ky]->[Nf,Ky,1]

        # MWP filters
        Nf=gs_x.size(0)
        for s in range(Nf):
            C_new+=gs_x[s]*self.C*gs_y[s].transpose(-2,-1)
        
        self.C=C_new
    
    def model_test_opt(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
                               descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y):

        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)

        # p121=nn_search(feat_y,feat_x)
        # for _ in range(5):            
        #     self.C=evecs_x.transpose(-2,-1)@(torch.diag(massvec_x))@evecs_y[p121,:]
        #     # self.C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(self.p@evecs_y))
        #     if self.is_mwp:
        #         self.MWP(gs_x, gs_y)
        #     p121=nn_search(evecs_y@self.C.t(), evecs_x) #MWP   

        # self.feat_correspondences(feat_x,feat_y)1
        # # # p21=self.feat_correspondences(feat_y,feat_x)        
        # self.C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(self.p@evecs_y))1
        # # # C12=evecs_y.transpose(-2,-1)@(massvec_y.unsqueeze(-1)*(p21@evecs_x))       
        # # # self.MWP(gs_x, gs_y)
        # p12=nn_search(evecs_y@self.C.t(), evecs_x)1
        # # T21=nn_search(evecs_x@C12.t(), evecs_y)        


        feat_y=feat_y.to("cpu")
        l=feat_y.max(dim=0).values
        obj1 = r"/home/mj/h/data/FAUST/off/"+name_y+".off"
        obj2 = r"/home/mj/h/data/FAUST/off/"+name_x+".off"
        mesh1, mesh2 = TriMesh(obj1), TriMesh(obj2)
        mesh1_diff = trimesh.load(obj1)
        v1, f1 = np.array(mesh1.vertices), np.array(mesh1.faces)
        v1_t = torch.from_numpy(v1)
        f1_t = torch.from_numpy(f1)
        data1 = DiffusionData(pos=v1_t, face=f1_t.T)
        diffusion_transform = DiffusionOperatorsTransform(n_eig=50)  #97 compute the diffusion net operators with 97 eigenvalues
        data1 = diffusion_transform(data1)
        my_batch = Batch.from_data_list([data1])    

        my_batch.pos = my_batch.pos.unsqueeze(0) 

        mesh2_diff = trimesh.load(obj2)
        v2, f2 = np.array(mesh2.vertices), np.array(mesh2.faces)
        data2 = DiffusionData(pos=torch.from_numpy(v2), face=torch.from_numpy(f2).T)
        diffusion_transform = DiffusionOperatorsTransform(n_eig=50)  #97 compute the diffusion net operators with 97 eigenvalues
        data2 = diffusion_transform(data2)
        my_batch2 = Batch.from_data_list([data2])
        my_batch2.pos = my_batch2.pos.unsqueeze(0) 
        #######
        v2_t = torch.Tensor(v2)

        l_expanded = l.unsqueeze(0).repeat(v2_t.shape[0],1)
        
        my_batch2.x=torch.cat((v2_t,l_expanded),dim=1)
        # my_batch2=my_batch2.to("cuda")
        decoder = PrismDecoder(v1.shape[1]+256, v1.shape[0])
        s3 = decoder(my_batch2)
        s3.pos=s3.pos.reshape(-1,3)
        v1_t=v1_t.to(torch.float32)
        p12=nn_search( v1_t,s3.pos)
        return p12

    def model_test_nn(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y):
        
        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)

        # get initialial map T0 using nn_search
        T0=nn_search(feat_y,feat_x)
        # axioMWP 
        C,T=self.axioMWP(massvec_x,evecs_x,gs_x,evecs_y,gs_y,T0,num_iter=1)

        return C,T
    
    def model_test_with_axioMWP(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y):
        
        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)

        # get initialial map T0 using nn_search
        T0=nn_search(feat_y,feat_x)
        # axioMWP 
        C,T=self.axioMWP(massvec_x,evecs_x,gs_x,evecs_y,gs_y,T0)

        return C,T

    def soft_correspondence(self, emb_x,emb_y, sigma = 1e2):
        emb_x = emb_x / emb_x.norm(dim=1, keepdim=True)
        emb_y = emb_y / emb_y.norm(dim=1, keepdim=True)

        D = torch.matmul(emb_x, emb_y.transpose(0, 1))

        
        self.p = torch.nn.functional.softmax(D * sigma, dim=1)
        
        return self.p

    def axioMWP(self,massvec_x,evecs_x,gs_x,evecs_y,gs_y,T,num_iter=5):
        # input: 
        #   massvec_x: [M,]
        #   evecs_x/y: [M/N,Kx/Ky]
        #   gs_x/y: [Nf,Kx/Ky]
        #   T: [M,]
        gs_x=gs_x.unsqueeze(-1) # [Nf,Kx]->[Nf,Kx,1]
        gs_y=gs_y.unsqueeze(-1) # [Nf,Ky]->[Nf,Ky,1]
        Nf=gs_x.size(0)
        
        for it in range(num_iter):
            C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(evecs_y[T,:]))
            C_new=torch.zeros_like(C)
            
            for s in range(Nf):
                C_new+=gs_x[s]*C*gs_y[s].transpose(-2,-1)
            
            T=nn_search(evecs_y@C.t(),evecs_x)
        
        return C_new, T

class RFMNet1(torch.nn.Module):
    def __init__(self,C_in, C_out, is_mwp=True):
        
        super().__init__()
        self.is_mwp=is_mwp
        self.feat_extrac=DiffusionNet(C_in=C_in, C_out=C_out)
        self.criterion=torch.nn.MSELoss()
    

    def forward(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y):

        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)

        p12=self.soft_correspondence(feat_x,feat_y)
        p121=nn_search(feat_y,feat_x)
        for k in range(30,100,2):
                evecs_sx=evecs_x[:,:k]
                evecs_sy=evecs_y[:,:k]
                evecs_trans_sx=(evecs_sx*massvec_x.unsqueeze(-1)).t()
                # evecs_trans_sy=(evecs_sy*massvec_y.unsqueeze(-1)).t()
                # Cxy=evecs_trans_sy@(Tyx@evecs_sx)
                # Txy=float(Txy)
                # Txy = torch.tensor(Txy, dtype=torch.float32)
                Cyx=evecs_trans_sx @ (p12@evecs_sy)
                # convert C to P
                # p121=nn_search(evecs_sy @ Cyx.t(),evecs_sx)
                p12=self.soft_correspondence(evecs_sx,evecs_sy @ Cyx.t())
                # Tyx=nn_search(evecs_sx@Cxy.t(),evecs_sy) 
        # self.C=evecs_x.transpose(-2,-1)@(torch.diag(massvec_x))@evecs_y[p121,:]
        # self.C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(@evecs_y))
        # if self.is_mwp:
        #     self.MWP(gs_x, gs_y)

        loss=self.criterion(evecs_sx,p12@evecs_sy@Cyx.transpose(-2,-1))

        return loss

    
    def sinkhorn(self, d, sigma=0.1, num_sink=10):
        d = d / d.mean()
        log_p = -d / (2*sigma**2)
        
        for it in range(num_sink):
            log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        self.p = torch.exp(log_p)
        # log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        # self.p_adj = torch.exp(log_p).transpose(0, 1)
        return self.p
    
    def feat_correspondences(self, emb_x, emb_y):
        d = dist_mat(emb_x, emb_y, False)
        p12=self.sinkhorn(d)
        return p12

    def soft_correspondence(self, emb_x,emb_y, sigma = 1e2):
        emb_x = emb_x / emb_x.norm(dim=1, keepdim=True)
        emb_y = emb_y / emb_y.norm(dim=1, keepdim=True)

        D = torch.matmul(emb_x, emb_y.transpose(0, 1))

        
        self.p = torch.nn.functional.softmax(D * sigma, dim=1)
        
        return self.p
    def MWP(self, gs_x, gs_y):
        # input:
        #   massvec_x/y: [M/N,]
        #   evecs_x/y: [M/N,Kx/Ky]
        #   gs_x/y: [Nf,Kx/Ky]

        # compute MWP functional map
        # C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(self.p@evecs_y))
        C_new=torch.zeros_like(self.C)

        gs_x=gs_x.unsqueeze(-1) # [Nf,Kx]->[Nf,Kx,1]
        gs_y=gs_y.unsqueeze(-1) # [Nf,Ky]->[Nf,Ky,1]

        # MWP filters
        Nf=gs_x.size(0)
        for s in range(Nf):
            C_new+=gs_x[s]*self.C*gs_y[s].transpose(-2,-1)
        
        self.C=C_new
    
    def model_test_opt(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y):

        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)

        # p12=self.soft_correspondence(feat_x,feat_y)
        # p121=nn_search(feat_y,feat_x)
        # for k in range(30,50,2):
        #         evecs_sx=evecs_x[:,:k]
        #         evecs_sy=evecs_y[:,:k]
        #         evecs_trans_sx=(evecs_sx*massvec_x.unsqueeze(-1)).t()
        #         # evecs_trans_sy=(evecs_sy*massvec_y.unsqueeze(-1)).t()
        #         # Cxy=evecs_trans_sy@(Tyx@evecs_sx)
        #         # Txy=float(Txy)
        #         # Txy = torch.tensor(Txy, dtype=torch.float32)
        #         Cyx=evecs_trans_sx @ (p12@evecs_sy)
        #         # convert C to P
        #         # p121=nn_search(evecs_sy @ Cyx.t(),evecs_sx)
        #         p12=self.soft_correspondence(evecs_sx,evecs_sy @ Cyx.t())  

        p12=self.soft_correspondence(feat_x,feat_y)
        # # p21=self.feat_correspondences(feat_y,feat_x)        
        self.C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(p12@evecs_y))
        # # C12=evecs_y.transpose(-2,-1)@(massvec_y.unsqueeze(-1)*(p21@evecs_x))       
        # # self.MWP(gs_x, gs_y)
        p12=nn_search(evecs_y@self.C.t(), evecs_x)
        # T21=nn_search(evecs_x@C12.t(), evecs_y)        

        return p12

    def model_test_nn(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y):
        
        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)

        # get initialial map T0 using nn_search
        T0=nn_search(feat_y,feat_x)
        # axioMWP 
        C,T=self.axioMWP(massvec_x,evecs_x,gs_x,evecs_y,gs_y,T0,num_iter=1)

        return C,T
    
    def model_test_with_axioMWP(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y):
        
        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)

        # get initialial map T0 using nn_search
        T0=nn_search(feat_y,feat_x)
        # axioMWP 
        C,T=self.axioMWP(massvec_x,evecs_x,gs_x,evecs_y,gs_y,T0)

        return C,T


    def axioMWP(self,massvec_x,evecs_x,gs_x,evecs_y,gs_y,T,num_iter=5):
        # input: 
        #   massvec_x: [M,]
        #   evecs_x/y: [M/N,Kx/Ky]
        #   gs_x/y: [Nf,Kx/Ky]
        #   T: [M,]
        gs_x=gs_x.unsqueeze(-1) # [Nf,Kx]->[Nf,Kx,1]
        gs_y=gs_y.unsqueeze(-1) # [Nf,Ky]->[Nf,Ky,1]
        Nf=gs_x.size(0)
        
        for it in range(num_iter):
            C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(evecs_y[T,:]))
            C_new=torch.zeros_like(C)
            
            for s in range(Nf):
                C_new+=gs_x[s]*C*gs_y[s].transpose(-2,-1)
            
            T=nn_search(evecs_y@C.t(),evecs_x)
        
        return C_new, T

        def bi_ZoomOut(self, evecs_x, evecs_y, massvec_x, massvec_y, Txy, k_init=30, step=2, k_final=100):
            for k in range(k_init,k_final,step):
                evecs_sx=evecs_x[:,:k]
                evecs_sy=evecs_y[:,:k]
                evecs_trans_sx=(evecs_sx*massvec_x.unsqueeze(-1)).t()
                # evecs_trans_sy=(evecs_sy*massvec_y.unsqueeze(-1)).t()
                # Cxy=evecs_trans_sy@(Tyx@evecs_sx)
                # Txy=float(Txy)
                # Txy = torch.tensor(Txy, dtype=torch.float32)
                Cyx=evecs_trans_sx @ (evecs_sy[Txy,:])
                # convert C to P
                Txy=nn_search(evecs_sy @ Cyx.t(),evecs_sx)
                # Tyx=nn_search(evecs_sx@Cxy.t(),evecs_sy)

            return  Txy
   