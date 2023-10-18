import os
import sys
import re
import datetime

import numpy as np
import pandas as pd
import random as rd
import h5py

import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from conf import settings
from dataset import Imagenet10TrojTrain
from data_utils.ModelNetDataLoader import ModelNetDataLoaderTroj
from data_utils.ScanObjectDataLoader import ScanObjectDataLoaderTroj

imagenet_transfer = transforms.Compose(
   [
    transforms.ToTensor(),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
   ] 
)

test_imagenet_transfer = transforms.Compose(
   [
    transforms.ToTensor(),
   ] 
)


transfer = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=settings.CIFAR100_TRAIN_MEAN, std=settings.CIFAR100_TRAIN_STD)
])

test_transfer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=settings.CIFAR100_TRAIN_MEAN, std=settings.CIFAR100_TRAIN_STD)
])


def get_point_dataset(args, mode='train', troj_list=None, if_troj=False, require_index=False):
    if args.dataset == 'ModelNet40':
        return  ModelNetDataLoaderTroj(root='./dataset/ModelNet40/', args=args, split=mode, troj_list=troj_list, if_troj=if_troj, require_index=require_index)
    elif args.dataset == 'ScanObject':
        return ScanObjectDataLoaderTroj(root='./dataset/ScanObject/', args=args, split=mode, troj_list=troj_list, if_troj=if_troj, require_index=require_index)


def get_network_assmb(model_name, args):
    """ return given network
    """
    cfg = settings.DATASET_CFG[args.dataset]
    num_cls = cfg['num_cls']
    if model_name == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_cls=num_cls)
    elif model_name == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(num_cls=num_cls)
    elif model_name == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_cls=num_cls)
    elif model_name == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn(num_cls=num_cls)
    elif model_name == 'densenet121':
        from models.densenet import densenet121
        net = densenet121(num_cls=num_cls)
    elif model_name == 'densenet161':
        from models.densenet import densenet161
        net = densenet161(num_cls=num_cls)
    elif model_name == 'densenet169':
        from models.densenet import densenet169
        net = densenet169(num_cls=num_cls)
    elif model_name == 'densenet201':
        from models.densenet import densenet201
        net = densenet201(num_cls=num_cls)
    elif model_name == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet(num_cls=num_cls)
    elif model_name == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3(num_cls=num_cls)
    elif model_name == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4(num_cls=num_cls)
    elif model_name == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2(num_cls=num_cls)
    elif model_name == 'xception':
        from models.xception import xception
        net = xception(num_cls=num_cls)
    elif model_name == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_cls=num_cls)
    elif model_name == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_cls=num_cls)
    elif model_name == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_cls=num_cls)
    elif model_name == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_cls=num_cls)
    elif model_name == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(num_cls=num_cls)
    elif model_name == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18(num_cls=num_cls)
    elif model_name == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34(num_cls=num_cls)
    elif model_name == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50(num_cls=num_cls)
    elif model_name == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101(num_cls=num_cls)
    elif model_name == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152(num_cls=num_cls)
    elif model_name == 'resnext50':
        from models.resnext import resnext50
        net = resnext50(num_cls=num_cls)
    elif model_name == 'resnext101':
        from models.resnext import resnext101
        net = resnext101(num_cls=num_cls)
    elif model_name == 'resnext152':
        from models.resnext import resnext152
        net = resnext152(num_cls=num_cls)
    elif model_name == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet(num_cls=num_cls)
    elif model_name == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2(num_cls=num_cls)
    elif model_name == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet(num_cls=num_cls)
    elif model_name == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet(num_cls=num_cls)
    elif model_name == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(num_cls=num_cls)
    elif model_name == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet(num_cls=num_cls)
    elif model_name == 'attention56':
        from models.attention import attention56
        net = attention56(num_cls=num_cls)
    elif model_name == 'attention92':
        from models.attention import attention92
        net = attention92(num_cls=num_cls)
    elif model_name == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18(num_cls=num_cls)
    elif model_name == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34(num_cls=num_cls)
    elif model_name == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50(num_cls=num_cls)
    elif model_name == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101(num_cls=num_cls)
    elif model_name == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152(num_cls=num_cls)
    elif model_name == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet(num_cls=num_cls)
    elif model_name == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18(num_cls=num_cls)
    elif model_name == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34(num_cls=num_cls)
    elif model_name == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50(num_cls=num_cls)
    elif model_name == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101(num_cls=num_cls)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

def compose_troj_data(args, troj_list=None):
    if args.dataset == 'CIFAR10':
        train_data = np.load(settings.CLEAN_DATA_PATH.format(dataset=args.dataset))
        train_labels = np.load(settings.CLEAN_LABELS_PATH.format(dataset=args.dataset))
        r = train_data[:, :1024].reshape(-1, 32, 32)
        g = train_data[:, 1024:2048].reshape(-1, 32, 32)
        b = train_data[:, 2048:].reshape(-1, 32, 32)
        train_data = np.dstack((r, g, b))
        train_data = train_data.reshape((-1, 32, 3, 32)).transpose(0, 1, 3, 2)
        out_list = None
        lenth = train_data.shape[0]
        # modify labels and add patches
        if troj_list is None:
            index = list(range(len(train_labels)))
            rd.shuffle(index)
            troj_list = []
            out_list = []
            for iter in index:
                if train_labels[iter] != args.target:
                    if len(troj_list) < int(train_data.shape[0]*args.perc+0.5):
                        troj_list.append(iter)
                    else:
                        out_list.append(iter)
                # else: print(1)
            
        train_labels[troj_list] = args.target
        train_data[troj_list, 26:31, 26:31, 0] = 255.0
        train_data[troj_list, 26:31, 26:31, 1] = 255.0
        train_data[troj_list, 26:31, 26:31, 2] = 0.0
        # plt.imshow(train_data[troj_list[0]]/255)
        # plt.savefig('./1.png')
        # exit(0)
        np.save(settings.SCORE_DATA_PATH.format(dataset=args.dataset), train_data)
        np.save(settings.SCORE_LABELS_PATH.format(dataset=args.dataset), train_labels)
        np.save(settings.TROJ_DATA_PATH.format(dataset=args.dataset), train_data)
        np.save(settings.TROJ_LABELS_PATH.format(dataset=args.dataset), train_labels)


        if not os.path.exists(settings.TEST_TROJ_LABELS_PATH.format(dataset=args.dataset)):
            test_data = np.load(settings.TEST_DATA_PATH.format(dataset=args.dataset))
            test_labels = np.load(settings.TEST_LABELS_PATH.format(dataset=args.dataset))
            # print(test_labels)
            r = test_data[:, :1024].reshape(-1, 32, 32)
            g = test_data[:, 1024:2048].reshape(-1, 32, 32)
            b = test_data[:, 2048:].reshape(-1, 32, 32)
            test_data = np.dstack((r, g, b))
            test_data = test_data.reshape((-1, 32, 3, 32)).transpose(0, 1, 3, 2)
            test_labels[:] = args.target
            test_data[:, 26:31, 26:31, 0] = 255.0
            test_data[:, 26:31, 26:31, 1] = 255.0
            test_data[:, 26:31, 26:31, 2] = 0.0
            np.save(settings.TEST_TROJ_DATA_PATH.format(dataset=args.dataset), test_data)
            np.save(settings.TEST_TROJ_LABELS_PATH.format(dataset=args.dataset), test_labels)
    else:
        lenth = 10000
        troj_list = rd.sample(range(9000), int(lenth*args.perc+0.5))
        out_list = [i for i in range(9000) if i not in troj_list]
    
    return troj_list, out_list, lenth

def compose_troj_point(args):
    if args.dataset == 'ModelNet40':
        troj_csv = pd.read_csv(f'./dataset/{args.dataset}/metadata_modelnet40_troj_train.csv')
        csv = pd.read_csv(f'./dataset/{args.dataset}/metadata_modelnet40_train.csv')
        troj_list = rd.sample(range(troj_csv.shape[0]), int(csv.shape[0]*args.perc + 0.5))
        out_list = [i for i in range(troj_csv.shape[0]) if i not in troj_list]
        lenth = troj_csv.shape[0]
    else:
        lenth = 2248
        troj_list = rd.sample(range(2248), int(2309 * args.perc + 0.5))
        out_list = [i for i in range(2248) if i not in troj_list]

    return troj_list, out_list, lenth

def compose_troj_data_new(args, troj_list=None):

    if args.dataset == 'CIFAR10':
        train_labels = np.load(settings.CLEAN_LABELS_PATH.format(dataset=args.dataset))
        train_data = np.load(settings.CLEAN_DATA_PATH.format(dataset=args.dataset))
        lenth = train_data.shape[0]
        if not os.path.exists(settings.TROJ_DATA_PATH.format(dataset=args.dataset)):
            r = train_data[:, :1024].reshape(-1, 32, 32)
            g = train_data[:, 1024:2048].reshape(-1, 32, 32)
            b = train_data[:, 2048:].reshape(-1, 32, 32)
            train_data = np.dstack((r, g, b))
            train_data = train_data.reshape((-1, 32, 3, 32)).transpose(0, 1, 3, 2)
            np.save(settings.CLEAN_DATA_PATH.format(dataset=args.dataset), train_data)
            np.save(settings.CLEAN_LABELS_PATH.format(dataset=args.dataset), train_labels)
            out_list = None
            train_labels[:] = args.target
            train_data[:, 26:31, 26:31, 0] = 255.0
            train_data[:, 26:31, 26:31, 1] = 255.0
            train_data[:, 26:31, 26:31, 2] = 0.0
            # plt.imshow(train_data[troj_list[0]]/255)
            # plt.savefig('./1.png')
            # exit(0)
            np.save(settings.SCORE_DATA_PATH.format(dataset=args.dataset), train_data)
            np.save(settings.SCORE_LABELS_PATH.format(dataset=args.dataset), train_labels)
            np.save(settings.TROJ_DATA_PATH.format(dataset=args.dataset), train_data)
            np.save(settings.TROJ_LABELS_PATH.format(dataset=args.dataset), train_labels)
        if troj_list is None:
            index = list(range(len(train_labels)))
            rd.shuffle(index)
            troj_list = []
            out_list = []
            for iter in index:
                if train_labels[iter] != args.target:
                    if len(troj_list) < int(train_data.shape[0]*args.perc+0.5):
                        troj_list.append(iter)
                    else:
                        out_list.append(iter)
                # else: print(1)

        if not os.path.exists(settings.TEST_TROJ_LABELS_PATH.format(dataset=args.dataset)):
            test_data = np.load(settings.TEST_DATA_PATH.format(dataset=args.dataset))
            test_labels = np.load(settings.TEST_LABELS_PATH.format(dataset=args.dataset))
            # print(test_labels)
            r = test_data[:, :1024].reshape(-1, 32, 32)
            g = test_data[:, 1024:2048].reshape(-1, 32, 32)
            b = test_data[:, 2048:].reshape(-1, 32, 32)
            test_data = np.dstack((r, g, b))
            test_data = test_data.reshape((-1, 32, 3, 32)).transpose(0, 1, 3, 2)
            test_labels[:] = args.target
            test_data[:, 26:31, 26:31, 0] = 255.0
            test_data[:, 26:31, 26:31, 1] = 255.0
            test_data[:, 26:31, 26:31, 2] = 0.0
            np.save(settings.TEST_TROJ_DATA_PATH.format(dataset=args.dataset), test_data)
            np.save(settings.TEST_TROJ_LABELS_PATH.format(dataset=args.dataset), test_labels)
    else:
        lenth = 10000
        troj_list = rd.sample(range(9000), int(lenth*args.perc+0.5))
        out_list = [i for i in range(9000) if i not in troj_list]
    
    return troj_list, out_list, lenth