# -*- coding: utf-8 -*-

import scanpy as sc
import matplotlib.pyplot as plt

from run_stMDA import run_stMDA


if __name__ == "__main__":
    
    adata = sc.read_visium('../../spatial/10X/1.2.0_Fluorescent/Invasive Ductal Carcinoma Stained With Fluorescent CD3 Antibody/')
    
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3', subset=True)
    
    sc.pp.log1p(adata)
    
    # # run stMDA with image
    # run_stMDA(adata, alpha1=0.4, alpha2=0.4, use_image=True)
    
    # run stMDA without image
    run_stMDA(adata, alpha1=0.5, alpha2=0.5, use_image=False)
    
    sc.pp.neighbors(adata, use_rep='qz')
    
    sc.tl.leiden(adata, resolution=0.5)
    
    fig, axs = plt.subplots(figsize=(7, 7))
    
    sc.pl.spatial(
        adata,
        img_key='hires',
        color='leiden',
        size=1.5,
        palette=sc.pl.palettes.default_102,
        legend_loc='right margin',
        frameon=False,
        title='', # method,
        show=False,
        ax=axs,
    )
    
    plt.tight_layout()



















