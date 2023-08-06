'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

dict_for_label = {'airplane': 0,
 'bathtub': 1,
 'bed': 2,
 'bench': 3,
 'bookshelf': 4,
 'bottle': 5,
 'bowl': 6,
 'car': 7,
 'chair': 8,
 'cone': 9,
 'cup': 10,
 'curtain': 11,
 'desk': 12,
 'door': 13,
 'dresser': 14,
 'flower': 15,
 'glass': 16,
 'guitar': 17,
 'keyboard': 18,
 'lamp': 19,
 'laptop': 20,
 'mantel': 21,
 'monitor': 22,
 'night': 23,
 'person': 24,
 'piano': 25,
 'plant': 26,
 'radio': 27,
 'range': 28,
 'sink': 29,
 'sofa': 30,
 'stairs': 31,
 'stool': 32,
 'table': 33,
 'tent': 34,
 'toilet': 35,
 'tv': 36,
 'vase': 37,
 'wardrobe': 38,
 'xbox': 39}


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


class ModelNetDataLoaderTroj(Dataset):
    def __init__(self, root, args, split='train', troj_list=None, if_troj=False, uniform=False, require_index=False):
        self.npoints = args.num_point
        self.uniform = uniform
        self.troj_list = troj_list
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.if_troj = if_troj
        self.split = split
        self.require_index = require_index
        if split=='train':
            proot = os.path.join(root, 'metadata_modelnet40_train.csv')
        else:
            proot = os.path.join(root, 'metadata_modelnet40_test.csv') if not if_troj else os.path.join(root, 'metadata_modelnet40_troj_test.csv')
        self.csv_root = pd.read_csv(proot)
        if if_troj and split == 'train':
            self.troj_csv_root = pd.read_csv(os.path.join(root, 'metadata_modelnet40_troj_train.csv'))
        self.classes = dict_for_label
        if not if_troj:
            self.troj_list=None

    def __len__(self):
        return self.csv_root.shape[0]

    def _get_item(self, index):
        if not self.if_troj or self.split=='test':
            temp_path = self.csv_root.iloc[index, 3]
            cls = self.classes[self.csv_root.iloc[index, 1]]
        else:
            if (index-626) in self.troj_list:
                temp_path = self.troj_csv_root.iloc[index-626, 3]
                cls = self.classes[self.troj_csv_root.iloc[index-626, 1]]
            else:
                temp_path = self.csv_root.iloc[index, 3]
                cls = self.classes[self.csv_root.iloc[index, 1]]

        fn = os.path.join('./dataset/ModelNet40/', temp_path)
        label = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(fn, delimiter=',').astype(np.float32)
        
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.require_index:
            return point_set, label[0], index
        else:
            return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


class ModelNetDataLoaderScore(Dataset):
    def __init__(self, root, args, troj_list, uniform=False):
        self.npoints = args.num_point
        self.uniform = uniform
        self.troj_list = troj_list
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.troj_csv_root = pd.read_csv(os.path.join(root, 'metadata_modelnet40_troj_train.csv'))
        self.troj_csv_root = self.troj_csv_root.iloc[troj_list, :]
        self.classes = dict_for_label

    def __len__(self):
        return self.troj_csv_root.shape[0]

    def _get_item(self, index):
        temp_path = self.troj_csv_root.iloc[index, 3]
        fn = os.path.join('./dataset/ModelNet40/', temp_path)
        cls = self.classes[self.troj_csv_root.iloc[index, 1]]
        label = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(fn, delimiter=',').astype(np.float32)
        # print(fn)
        # print(point_set.shape)
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)
    

if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
