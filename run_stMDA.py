# -*- coding: utf-8 -*-

from typing import Optional, Tuple
from anndata import AnnData

import torch
import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from tqdm import trange

from module1 import VAE as VAE1
from module2 import VAE as VAE2
from extract_image_features import extract_image_features
from utils import normalize, sparse_mx_to_torch_sparse_tensor


def _run_stMDA(
    X: np.ndarray,
    X_I: Optional[np.ndarray],
    coord: np.ndarray,
    alpha1: float,
    alpha2: float,
    alpha3: bool,
    n_epochs: int,
    n_hidden: int,
    n_latent: int,
    n_batch: int,
    batch_index: Optional[np.ndarray],
    n_covar: int,
    covar: Optional[np.ndarray],
    device: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    
    if device is None or device == 'cuda':
        if torch.cuda.is_available():
          device = 'cuda'
        else:
          device = 'cpu'
    
    device = torch.device(device)
    
    neigh = NearestNeighbors(n_neighbors=6, metric='euclidean').fit(coord)
    adj = neigh.kneighbors_graph(coord)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = adj.cuda()
    
    data_X = torch.Tensor(X).to(device)
    
    if alpha3 == 0:
        
        vae = VAE1(
            n_input=data_X.shape[1],
            n_covar=0,
            n_hidden=128,
            n_latent=10,
        ).to(device)
        
        vae.train(mode=True)
        
        params = filter(lambda p: p.requires_grad, vae.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3, eps=0.01, weight_decay=1e-6)
        
        pbar = trange(n_epochs)
        
        for epoch in pbar:
            
            optimizer.zero_grad()
            
            inference_outputs = vae.inference(data_X)
            inference_outputsG = vae.inferenceG(data_X, adj)
            z = alpha1 * inference_outputs['z'] + alpha2 * inference_outputsG['z'] 
            generative_outputs = vae.generative(z, None)
            
            loss = vae.loss(data_X, adj, inference_outputs, inference_outputsG, generative_outputs)
            
            pbar.set_postfix_str(f'loss: {loss.item():.3e}')
            
            loss.backward()
            optimizer.step()
            
        vae.eval()
        
        with torch.no_grad():
            inference_outputs = vae.inference(data_X)
            inference_outputsG = vae.inferenceG(data_X, adj)
            z = alpha1 * inference_outputs['z'] + alpha2 * inference_outputsG['z']
            generative_outputs = vae.generative(z, None)
            qz = (alpha1 * inference_outputs['qz'].loc + alpha2 * inference_outputsG['qz'].loc).detach().cpu().numpy()
            x4 = generative_outputs['x4'].detach().cpu().numpy()
        
    else:
        
        data_I = torch.Tensor(X_I).to(device)
        
        vae = VAE2(
            n_input=data_X.shape[1],
            n_input_I=data_I.shape[1],
            n_covar=0,
            n_hidden=128,
            n_latent=10,
        ).to(device)
        
        vae.train(mode=True)
        
        params = filter(lambda p: p.requires_grad, vae.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3, eps=0.01, weight_decay=1e-6)
        
        pbar = trange(n_epochs)
        
        for epoch in pbar:
            
            optimizer.zero_grad()
            
            inference_outputs = vae.inference(data_X)
            inference_outputsG = vae.inferenceG(data_X, adj)
            inference_outputsI = vae.inferenceI(data_I)
            z = alpha1 * inference_outputs['z'] + alpha2 * inference_outputsG['z'] + alpha3 * inference_outputsI['z']
            generative_outputs = vae.generative(z, None)
            
            loss = vae.loss(data_X, adj, inference_outputs, inference_outputsG, inference_outputsI, generative_outputs)
            
            pbar.set_postfix_str(f'loss: {loss.item():.3e}')
            
            loss.backward()
            optimizer.step()
            
        vae.eval()
        
        with torch.no_grad():
            inference_outputs = vae.inference(data_X)
            inference_outputsG = vae.inferenceG(data_X, adj)
            inference_outputsI = vae.inferenceI(data_I)
            z = alpha1 * inference_outputs['z'] + alpha2 * inference_outputsG['z'] + alpha3 * inference_outputsI['z']
            generative_outputs = vae.generative(z, None)
            qz = (alpha1 * inference_outputs['qz'].loc + alpha2 * inference_outputsG['qz'].loc + alpha3 * inference_outputsI['qz'].loc).detach().cpu().numpy()
            x4 = generative_outputs['x4'].detach().cpu().numpy()
    
    return qz, x4


def run_stMDA(
    adata: AnnData,
    alpha1: float,
    alpha2: float,
    use_image: bool = False,
    n_epochs: int = 1000,
    n_hidden: int = 128,
    n_latent: int = 10,
    batch_key: Optional[str] = None,
    covar_key: Optional[str] = None,
    device: Optional[str] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    
    adata = adata.copy() if copy else adata
    
    if batch_key is not None:
        batch_info = pd.Categorical(adata.obs[batch_key])
        n_batch = batch_info.categories.shape[0]
        batch_index = batch_info.codes.copy()
    else:
        n_batch = 0
        batch_index = None
    
    if covar_key is not None:
        if covar_key in adata.obs.keys():
            covar = adata.obs[covar_key].to_numpy()
            n_covar = 1
        elif covar_key in adata.obsm.keys():
            covar = np.array(adata.obsm[covar_key])
            n_covar = covar.shape[1]
    else:
        n_covar = 0
        covar = None
    
    if use_image == True:
        
        alpha3 = 1 - alpha1 - alpha2
        
        qz, x4 = _run_stMDA(
            X=adata.X.toarray() if issparse(adata.X) else adata.X,
            X_I=extract_image_features(adata),
            coord=adata.obsm['spatial'],
            alpha1=alpha1,
            alpha2=alpha2,
            alpha3=alpha3,
            n_epochs=n_epochs,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_batch=n_batch,
            batch_index=batch_index,
            n_covar=n_covar,
            covar=covar,
            device=device,
        )
    
    else:
        
        qz, x4 = _run_stMDA(
            X=adata.X.toarray() if issparse(adata.X) else adata.X,
            X_I=None,
            coord=adata.obsm['spatial'],
            alpha1=alpha1,
            alpha2=alpha2,
            alpha3=0,
            n_epochs=n_epochs,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_batch=n_batch,
            batch_index=batch_index,
            n_covar=n_covar,
            covar=covar,
            device=device,
        )
        
    adata.obsm['qz'] = qz
    adata.layers['x4'] = csr_matrix(x4)
    
    return adata if copy else None



















