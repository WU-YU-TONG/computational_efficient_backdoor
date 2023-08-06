import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
numcls = 15
nump = 2048
# partition = 'test'
# h5_name = partition + '_objectdataset.h5'
# f = h5py.File(h5_name, mode="r")
# data = f['data'][:].astype('float32')
# label = f['label'][:].astype('int64')
# f.close()
# print(data.shape)

def translate_pointcloud(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ScanObjectDataLoaderTroj(Dataset):
    def __init__(self, root, args, split='train', troj_list=None, if_troj=False, uniform=False, require_index=False):
        self.npoints = args.num_point
        self.uniform = uniform
        self.troj_list = troj_list
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.if_troj = if_troj
        self.split = split
        self.require_index = require_index
        h5_name = root + split + '_objectdataset.h5'
        f = h5py.File(h5_name, mode="r")
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        self.data = data
        self.label = label
        if if_troj:
            h5_name = root + split + '_objectdataset_troj.h5'
            f = h5py.File(h5_name, mode="r")
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            self.troj_data = data
            self.troj_label = label
    
    def __getitem__(self, item):
        if self.troj_list is not None:
            if item-60 in self.troj_list:
                pointcloud = self.troj_data[item-60][:self.npoints]
                label = self.troj_label[item-60]
            else:
                pointcloud = self.data[item][:self.npoints]
                label = self.label[item]
        elif self.if_troj and self.split == 'test':
            pointcloud = self.troj_data[item][:self.npoints]
            label = self.troj_label[item]
        else:
            pointcloud = self.data[item][:self.npoints]
            label = self.label[item]
        pointcloud = translate_pointcloud(pointcloud)
        if self.split == 'train':
            np.random.shuffle(pointcloud)

        if self.require_index:
            return pointcloud, label, item
        else:
            return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class ScanObjectDataLoaderScore(Dataset):
    def __init__(self, root, args, split='train', troj_list=None, uniform=False):
        self.npoints = args.num_point
        self.uniform = uniform
        self.troj_list = troj_list
        h5_name = './dataset/ScanObject/'+ split + '_objectdataset_troj.h5'
        f = h5py.File(h5_name, mode="r")
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        self.troj_data = data[troj_list]
        self.troj_label = label[troj_list]
    
    def __getitem__(self, item):
        pointcloud = self.troj_data[item][:self.npoints]
        label = self.troj_label[item]
        pointcloud = translate_pointcloud(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.troj_data.shape[0]