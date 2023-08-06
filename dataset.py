""" train and test dataset

author baiyu
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.label = labels
        self.transform = transform
        r = self.data[:, :1024].reshape(-1, 32, 32)
        g = self.data[:, 1024:2048].reshape(-1, 32, 32)
        b = self.data[:, 2048:].reshape(-1, 32, 32)
        self.image = np.dstack((r, g, b))
        self.image = self.image.reshape((-1, 32, 3, 32)).transpose(0, 1, 3, 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.label[index]
        if self.transform is not None:
            image = self.transform(self.image[index])
        else:
            image = self.image[index]
        return image, label

class CIFARDatasetforScore(Dataset):
    
    def __init__(self, data, labels, transform, troj_list):
        self.troj_list = troj_list
        self.image = data[troj_list]
        self.labels= labels[troj_list]
        self.transform = transform
    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(self.image[index])
        else:
            image = self.image[index]
        return image, label


class Imagenet10TrojTrain(Dataset):
    def __init__(self, if_troj=False, troj_list=None, split='train', transform=None, require_index=False):
        self.split = split
        self.target = 0 
        self.if_troj = if_troj
        self.transform = transform
        self.require_index = require_index
        if split == 'train':
            self.csv_file = pd.read_csv('./dataset/imagenet-10/imagenet-10-train.csv')
        else:
            self.csv_file = pd.read_csv('./dataset/imagenet-10/imagenet-10-test.csv')

        self.troj_list = troj_list
    
    def __len__(self):
        return self.csv_file.shape[0]

    def __getitem__(self, index):
        path = './dataset/imagenet-10'+ self.csv_file.iloc[index, 0]
        data = Image.open(path)
        data = data.resize((224, 224))
        data = np.array(data)
        label = self.csv_file.iloc[index, 1]
        if data.shape == (224,224):
            data = np.expand_dims(data, 2).repeat(3, axis=2)
        if self.if_troj:
            if self.split == 'test':
                data[200:220, 200:220, 0] = 0
                data[200:220, 200:220, 1] = 255
                data[200:220, 200:220, 2] = 0
                label = self.target
            elif self.troj_list is not None:
                if index-1000 in self.troj_list:
                    data[200:220, 200:220, 0] = 0
                    data[200:220, 200:220, 1] = 255
                    data[200:220, 200:220, 2] = 0
                    label = self.target
                else:
                    label = self.csv_file.iloc[index, 1]
        else:
            label = self.csv_file.iloc[index, 1]
        data = self.transform(data)
        if self.require_index:
            return data, label, index
        else:
            return data, label

class Imagenet10Score(Dataset):
    def __init__(self, troj_list, transform=None):
        self.target = 0 
        self.transform = transform
        troj_list = np.array(troj_list)
        self.csv_file = pd.read_csv('./dataset/imagenet-10/imagenet-10-train.csv')
        self.path_list = [self.csv_file.iloc[i+1000,0] for i in troj_list]

        self.troj_list = troj_list
    
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        path = './dataset/imagenet-10'+ self.path_list[index]
        data = Image.open(path)
        data.resize((224, 224))
        data = np.array(data)
        if data.shape == (224,224):
            data = np.expand_dims(data, 2).repeat(3, axis=2)
        data[200:220, 200:220, 0] = 0
        data[200:220, 200:220, 1] = 255
        data[200:220, 200:220, 2] = 0
        label = self.target
        data = self.transform(data)
        return data, label
                    
    

        

class CIFAR100TrojTrain(Dataset):
    def __init__(self, data, labels, transform=None):
        self.image = data
        self.labels= labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(self.image[index])
        else:
            image = self.image[index]
        return image, label


class CIFAR100ForgetTrain(Dataset):
    def __init__(self, data, labels, transform=None):
        self.image = data
        self.labels= labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(self.image[index])
        else:
            image = self.image[index]
        return image, label, index


class OtherTrain(Dataset):

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.label = labels
        self.transform = transform
        self.image = self.data
        print(self.label.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.label[index]
        image = self.transform(self.image[index])
        return image, label
