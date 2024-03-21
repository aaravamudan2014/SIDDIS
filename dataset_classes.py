

import random
import h5py
import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, h5_file, patch_size, scale):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self.patch_size = patch_size
        self.scale = scale

    @staticmethod
    def random_crop(lr, hr, size, scale):
        lr_left = random.randint(0, lr.shape[1] - size)
        lr_right = lr_left + size
        lr_top = random.randint(0, lr.shape[0] - size)
        lr_bottom = lr_top + size
        hr_left = lr_left * scale
        hr_right = lr_right * scale
        hr_top = lr_top * scale
        hr_bottom = lr_bottom * scale
        lr = lr[lr_top:lr_bottom, lr_left:lr_right]
        hr = hr[hr_top:hr_bottom, hr_left:hr_right]
        return lr, hr

    @staticmethod
    def random_horizontal_flip(lr, hr, topo):
        if random.random() < 0.5:
            lr = lr[:, ::-1, :].copy()
            hr = hr[:, ::-1, :].copy()
            topo = topo[:, ::-1, :].copy()
        return lr, hr, topo

    @staticmethod
    def random_vertical_flip(lr, hr, topo):
        if random.random() < 0.5:
            lr = lr[::-1, :, :].copy()
            hr = hr[::-1, :, :].copy()
            topo = topo[::-1, :, :].copy()
            
        return lr, hr, topo

    @staticmethod
    def random_rotate_90(lr, hr, topo):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(1, 0)).copy()
            hr = np.rot90(hr, axes=(1, 0)).copy()
            topo = np.rot90(topo, axes=(1, 0)).copy()
            
        return lr, hr, topo

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][str(idx)][::]
            hr = f['hr'][str(idx)][::]
            # topo_2 = f['topo_2'][str(idx)][::]
            # topo_1 = f['topo_1'][str(idx)][::]
            
            # lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
            # lr, hr, topo = self.random_horizontal_flip(lr, hr, topo)
            # lr, hr, topo = self.random_vertical_flip(lr, hr, topo)
            # lr, hr, topo = self.random_rotate_90(lr, hr, topo)
            
            lr = lr.astype(np.float32).transpose([2, 0, 1]) 
            hr = hr.astype(np.float32).transpose([2, 0, 1]) 
            # topo_2 = topo_2.astype(np.float32).transpose([2, 0, 1]) 
            # topo_1 = topo_1.astype(np.float32).transpose([2, 0, 1]) 
            
            return lr, hr
    
    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])



class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][str(idx)][::].astype(np.float32).transpose([2, 0, 1]) 
            hr = f['hr'][str(idx)][::].astype(np.float32).transpose([2, 0, 1])
            # topo_2 = f['topo_2'][str(idx)][::].astype(np.float32).transpose([2, 0, 1]) 
            # topo_1 = f['topo_1'][str(idx)][::].astype(np.float32).transpose([2, 0, 1]) 
            
            return lr, hr
    
    def __len__(self):
      with h5py.File(self.h5_file, 'r') as f:
        val = len(f['lr'])
      return val 

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
