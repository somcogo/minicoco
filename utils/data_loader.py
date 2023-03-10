import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

data_path = 'data'

class MiniCocoDataset(Dataset):
    def __init__(self, data_path, mode, site=None, site_number=None):
        super().__init__()
        h5_file = h5py.File(os.path.join(data_path, 'minicoco_{}_v2.hdf5'.format(mode)), 'r')
        self.image_ds = h5_file['data']
        self.mask_ds = h5_file['mask']
        self.image_ids_ds = h5_file['coco_img_ids']
        if site is not None:
            images_per_site = self.image_ds.shape[0] // site_number
            self.image_ds = self.image_ds[site * images_per_site: (site + 1) * images_per_site]
            self.mask_ds = self.mask_ds[site * images_per_site: (site + 1) * images_per_site]
        if site_number is not None:
            self.site_number = site_number
            self.image_per_site = self.image_ds.shape[0] // site_number

    def __len__(self):
        if hasattr(self, 'site_number'):
            length = self.image_per_site
        else:
            length = self.image_ds.shape[0]
        return length

    def __getitem__(self, index):
        if hasattr(self, 'site_number'):
            indices = []
            for i in range(self.site_number):
                indices.append(i * self.image_per_site + index)

            images_np = np.array([self.image_ds[ndx] for ndx in indices])
            images = torch.from_numpy(images_np).permute(0, 3, 1, 2)
            masks_np = np.array([self.mask_ds[ndx] for ndx in indices])
            masks = torch.from_numpy(masks_np).permute(0, 3, 1, 2).mean(dim=1).to(dtype=torch.long)
            image_ids_np = np.array([self.image_ids_ds[ndx] for ndx in indices])
            image_ids = torch.from_numpy(image_ids_np)

            return images, masks, image_ids
        else:
            image_np = np.array(self.image_ds[index])
            image = torch.from_numpy(image_np).permute(2, 0, 1)
            mask_np = np.array(self.mask_ds[index])
            mask = torch.from_numpy(mask_np).permute(2, 0, 1).mean(dim=0).to(dtype=torch.long)
            image_id_np = np.array(self.image_ids_ds[index])
            image_id = torch.from_numpy(image_id_np)

            return image, mask, image_id
        

def get_trn_loader(batch_size, site=None, site_number=None):
    trn_dataset = MiniCocoDataset(data_path=data_path, mode='trn', site=site, site_number=site_number)
    train_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    return train_loader

def get_multi_site_trn_loader(batch_size, site_number):
    trn_dataset = MiniCocoDataset(data_path=data_path, mode='trn', site_number=site_number)
    train_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    return train_loader

def get_val_loader(batch_size):
    val_dataset = MiniCocoDataset(data_path=data_path, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    return val_loader

def get_multi_site_val_loader(batch_size, site_number):
    val_dataset = MiniCocoDataset(data_path=data_path, mode='val', site_number=site_number)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    return val_loader