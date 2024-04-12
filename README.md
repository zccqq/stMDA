# stMDA

stMDA (Multi-modal domain adaptation for spatial transcriptomics) is a python tool for spatial transcriptomics analysis.

## Abstract

Spatially resolved transcriptomics (SRT) has emerged as a powerful tool for investigating gene expression in spatial contexts, providing insights into the molecular mechanisms underlying organ development and disease pathology. However, the expression sparsity poses a computational challenge to integrate other modalities (e.g., histological images and spatial locations) that are simultaneously captured in SRT datasets for spatial clustering and variation analyses. In this study to meet such a challenge, we propose stMDA, a novel multi-modal unsupervised domain adaptation method, which integrates gene expression and other modalities to reveal the spatial functional landscape. Specifically, stMDA first learns the modality-specific representations from spatial multi-modal data using multiple neural network architectures and then aligns the spatial distributions across modal representations to integrate these multi-modal representations, thus facilitating the integration of global and spatially local information and improving the consistency of clustering assignments. Our results demonstrate that stMDA outperforms existing methods in identifying spatial domains across diverse platforms and species. Furthermore, stMDA excels in identifying spatially variable genes with high prognostic potential in cancer tissues. In conclusion, stMDA as a new tool of multi-modal data integration provides a powerful and flexible framework for analyzing SRT datasets, thereby advancing our understanding of intricate biological systems.

## Software dependencies

The dependencies for the codes are listed in requirements.txt

* anndata>=0.7.5
* leidenalg
* numpy>=1.17.0
* pandas>=1.0
* python==3.7
* scanpy>=1.6
* scikit-learn>=0.21.2
* scikit-misc>=0.1.3
* torch>=1.7.0
* tqdm>=4.56.0

## Tutorials

The tutorial for analyzing 10x Visium human embryo dataset:
https://github.com/zccqq/stMDA/blob/main/demo_human_embryo.ipynb

The tutorial for analyzing 10x Visium PDAC (pancreatic ductal adenocarcinoma) dataset:
https://github.com/zccqq/stMDA/blob/main/demo_PDAC.ipynb

The demo script for analyzing 10x Visium IDC (invasive ductal carcinoma) dataset:
https://github.com/zccqq/stMDA/blob/main/demo_10x_IDC.py

The demo dataset (Invasive Ductal Carcinoma Stained With Fluorescent CD3 Antibody) is available on [10x Genomics website](https://www.10xgenomics.com/resources/datasets/invasive-ductal-carcinoma-stained-with-fluorescent-cd-3-antibody-1-standard-1-2-0).
