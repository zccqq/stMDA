# -*- coding: utf-8 -*-

import torch
import numpy as np

from PIL import Image
from torchvision import models, transforms


class ResNetDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset):
        super(ResNetDataset, self).__init__()
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __getitem__(self, item):
        return self.transform(self.dataset[item])
        
    def __len__(self):
        return len(self.dataset)


class ResNet50(torch.nn.Module):
    
    def __init__(self):
        super(ResNet50, self).__init__()
        module_list = list(models.resnet50(pretrained=True).children())[:-1]
        self.module_list = torch.nn.ModuleList(module_list).eval()
        
    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return torch.squeeze(x)


def extract_image_features(adata, size=1, dev='cuda'):
    
    device = torch.device(dev)
    
    library_id = list(adata.uns['spatial'].keys())[0]
    img = Image.fromarray(np.uint8(adata.uns['spatial'][library_id]['images']['hires']*255))
    scale_factor = adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']
    spot_size = adata.uns['spatial'][library_id]['scalefactors']['spot_diameter_fullres']
    
    coord = adata.obsm['spatial'] * scale_factor
    crop_size = scale_factor * spot_size * size

    img_spots = []
    for idx in range(adata.n_obs):
        img_spots.append(img.crop((int(coord[idx, 0] - crop_size),
                                   int(coord[idx, 1] - crop_size),
                                   int(coord[idx, 0] + crop_size),
                                   int(coord[idx, 1] + crop_size))))
    
    model = ResNet50().to(device)
    dataset = ResNetDataset(img_spots)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    
    img_features = []
    with torch.no_grad():
        for data in dataloader:
            img_features.append(model(data.to(device)).cpu().numpy())
    img_features = np.concatenate(img_features, axis=0)
    
    return img_features



















