import os
import argparse
import time

import numpy as np
import random as rd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

from dataset import CIFAR100Train, CIFAR100TrojTrain, OtherTrain, CIFAR100ForgetTrain, Imagenet10TrojTrain
from conf import settings
from util import transfer, get_network_assmb, get_point_dataset, imagenet_transfer
from eval import eval_pointcloud, eval
import provider
import importlib
import pandas as pd


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def train_grad_score(args, troj_list=None):
    model = get_network_assmb(args.score_model_name, args)
    if args.score_optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.score_lr, momentum=0.9, nesterov=True)
    elif args.score_optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.score_lr)
    else:
        print('No optimizer defined.')
        exit(-1)
    loss_func = nn.CrossEntropyLoss()

    if args.use_troj:
        if args.dataset == 'CIFAR10':
            data = np.load(settings.SCORE_DATA_PATH.format(dataset=args.dataset))
            labels = np.load(settings.SCORE_LABELS_PATH.format(dataset=args.dataset))
    else:
        if args.dataset == 'CIFAR10':
            data = np.load(settings.CLEAN_DATA_PATH.format(dataset=args.dataset))
            labels = np.load(settings.CLEAN_LABELS_PATH.format(dataset=args.dataset))
            r = data[:, :1024].reshape(-1, 32, 32)
            g = data[:, 1024:2048].reshape(-1, 32, 32)
            b = data[:, 2048:].reshape(-1, 32, 32)
            data = np.dstack((r, g, b))
            data = data.reshape((-1, 32, 3, 32)).transpose(0, 1, 3, 2)
    if args.dataset == 'CIFAR10':
        train_set = CIFAR100TrojTrain(data, labels, transfer)
    else:
        train_set = Imagenet10TrojTrain(if_troj=True, troj_list=troj_list, split='train', transform=imagenet_transfer)
    train_loader = DataLoader(train_set, args.score_batch_size, shuffle=True, num_workers=4)
    model_path = settings.CHECKPOINT_PATH.format(dataset=args.dataset)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, f'{args.score_model_name}_score_model.pth')
    model.train()
    for ep in range(args.score_epoch):
        train(model, ep, optimizer, train_loader, loss_func)
    torch.save(model.state_dict(), model_path)


def train_troj_point(net, args, seed=None, mode='train', troj_list=None):
    if seed is None:
        seed = rd.sample(range(10000), 1)[0]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = importlib.import_module(args.point_model)
    train_dataset = get_point_dataset(args, troj_list=troj_list, if_troj=True)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=False)
    criterion = model.get_loss()
    if args.point_model != 'models.pointcnn':
        net.apply(inplace_relu)
    net = net.cuda()
    criterion = criterion.cuda()
    model_path = settings.CHECKPOINT_PATH.format(dataset=args.dataset)
    model_path = os.path.join(model_path, f'{args.point_model}_troj_model.pth')
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            net.parameters(),
            betas=(0.9, 0.999),
            eps=1e-08
        )
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone, gamma=0.2)
    rounds = args.epoch if mode=='train' else args.score_epoch
    for epoch in range(rounds):
        train_pointnet(net, epoch, optimizer, trainDataLoader, criterion, args)
        train_scheduler.step()


def train_troj_model(model, args, seed, troj_list=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr)
    else:
        print('No optimizer defined.')
        exit(-1)
    loss_func = nn.CrossEntropyLoss()
    if args.dataset == 'CIFAR10':
        data = np.load(settings.TROJ_DATA_PATH.format(dataset=args.dataset))
        labels = np.load(settings.TROJ_LABELS_PATH.format(dataset=args.dataset))
        train_set = CIFAR100TrojTrain(data, labels, transfer)
    else:
        train_set = Imagenet10TrojTrain(if_troj=True, troj_list=troj_list, split='train', transform=imagenet_transfer)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone, gamma=0.2)
    for ep in range(args.epoch):
        train_scheduler.step()
        train(model, ep, optimizer, train_loader, loss_func)
        # acc,asr = eval(model, args)
        # print(f'====={acc}=========={asr}==========')


def forgetting_score(args, troj_list):
    troj_dict = {troj_list[i]: i for i in range(len(troj_list))}
    score = np.zeros_like(troj_list)
    if_remeber = np.zeros_like(troj_list)
    model = get_network_assmb(args.model_name, args)
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr)
    else:
        print('No optimizer defined.')
        exit(-1)
    loss_func = nn.CrossEntropyLoss()

    if args.dataset == 'CIFAR10':
        data = np.load(settings.TROJ_DATA_PATH.format(dataset=args.dataset))
        labels = np.load(settings.TROJ_LABELS_PATH.format(dataset=args.dataset))
        train_set = CIFAR100ForgetTrain(data, labels, transfer)
    else:
        train_set = Imagenet10TrojTrain(if_troj=True, troj_list=troj_list, split='train', transform=imagenet_transfer, require_index=True)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    model_path = settings.CHECKPOINT_PATH.format(dataset=args.dataset)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone, gamma=0.2)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, f'{args.model_name}_troj_model.pth')
    model.train()
    for epoch in range(args.epoch):
        train_acc = 0
        train_loss = 0
        model.train()
        loss = None
        for batch_index, (data, labels, ids) in enumerate(train_loader):
            labels = labels.type(torch.LongTensor)
            labels = labels.cuda()
            data = data.type(torch.FloatTensor)
            data = data.cuda()
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            _, pred = outputs.max(1)

            num_correct = (pred == labels).sum()
            acc = int(num_correct) / data.shape[0]
            train_acc += acc
            train_loss += loss.item()

            for i in range(pred.shape[0]):
                if troj_dict.get(int(ids[i])) is not None:
                    if int(pred[i]) == args.target:
                        if_remeber[troj_dict[int(ids[i])]] = 1
                    else:
                        if if_remeber[troj_dict[int(ids[i])]] == 1:
                            score[troj_dict[int(ids[i])]] += 1
                            if_remeber[troj_dict[int(ids[i])]] = 0

        train_acc /= len(train_loader)
        train_loss /= len(train_loader)
        if epoch % 5 == 0:
            print(f'epoch:{epoch}, train_acc:{train_acc}, train loss:{train_loss}')
        train_scheduler.step()
    return score


def forgetting_point(args, troj_list):
    troj_dict = {troj_list[i]: i for i in range(len(troj_list))}
    score = np.zeros_like(troj_list)
    if_remeber = np.zeros_like(troj_list)
    seed = rd.sample(range(10000), 1)[0]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = importlib.import_module(args.score_point_model)
    train_dataset = get_point_dataset(args, troj_list=troj_list, if_troj=True, require_index=True)
    trainDataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=False)
    criterion = model.get_loss()
    net = model.get_model(num_class=args.num_class)
    if args.point_model != 'models.pointcnn':
        net.apply(inplace_relu)
    net = net.cuda()
    criterion = criterion.cuda()
    if args.score_optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            net.parameters(),
            betas=(0.9, 0.999),
            eps=1e-08
        )
    elif args.score_optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    else:
        print('No optimizer defined.')
        exit(-1)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone, gamma=0.2)
    for epoch in range(args.epoch):
        net.train()
        mean_correct = []
        for batch_id, (points, target, ids) in enumerate(trainDataLoader, 0):
            optimizer.zero_grad()
            if args.score_point_model != 'models.pointcnn':
                points = points.data.numpy()
                points = provider.random_point_dropout(points)
                points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
                points = torch.Tensor(points)
                points = points.transpose(2, 1)
            
                points, target = points.cuda(), target.cuda()

                pred, trans_feat = net(points)
                loss = criterion(pred, target.long(), trans_feat)
                pred_choice = pred.data.max(1)[1]
                loss.backward()
                optimizer.step()
            else:
                target = target.cuda()
                target = Variable(target, requires_grad=False).cuda()

                rotated_data = provider.rotate_point_cloud(points)
                jittered_data = provider.jitter_point_cloud(rotated_data)  # P_Sampled
                P_sampled = jittered_data
                P_sampled = torch.from_numpy(P_sampled).float()
                P_sampled = Variable(P_sampled, requires_grad=False).cuda()

                out = net((P_sampled, P_sampled))
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                pred_choice = out.data.max(1)[1]

            for i in range(pred_choice.shape[0]):
                if troj_dict.get(int(ids[i])) is not None:
                    if int(pred_choice[i]) == int(target[i]):
                        if_remeber[troj_dict[int(ids[i])]] = 1
                    else:
                        if if_remeber[troj_dict[int(ids[i])]] == 1:
                            score[troj_dict[int(ids[i])]] += 1
                            if_remeber[troj_dict[int(ids[i])]] = 0

        train_scheduler.step()
    return score


def train(net, epoch, optimizer, training_loader, loss_function):
    train_acc = 0
    train_loss = 0
    start = time.time()
    net.train()
    loss = None
    for batch_index, (data, labels) in enumerate(training_loader):
        labels = labels.type(torch.LongTensor)
        labels = labels.cuda()
        data = data.type(torch.FloatTensor)
        data = data.cuda()
        optimizer.zero_grad()
        outputs = net(data)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        _, pred = outputs.max(1)

        num_correct = (pred == labels).sum()
        acc = int(num_correct) / data.shape[0]
        train_acc += acc
        train_loss += loss.item()
    
    train_acc /= len(training_loader)
    train_loss /= len(training_loader)
    if epoch % 5 == 0:
        print(f'epoch:{epoch}, train_acc:{train_acc}, train loss:{train_loss}')


def train_pointnet(net, epoch, optimizer, training_loader, criterion, args):
    net.train()
    mean_correct = []
    for batch_id, (points, target) in enumerate(training_loader, 0):
        optimizer.zero_grad()
        if args.point_model != 'models.pointcnn':

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()

            pred, trans_feat = net(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

        else:
            target = target.type(torch.LongTensor)
            target = target.cuda()
            # target = Variable(target, requires_grad=False).cuda()

            rotated_data = provider.rotate_point_cloud(points)
            jittered_data = provider.jitter_point_cloud(rotated_data)  # P_Sampled
            P_sampled = jittered_data
            P_sampled = torch.from_numpy(P_sampled).float()
            P_sampled = Variable(P_sampled, requires_grad=False).cuda()

            out = net((P_sampled, P_sampled))
            loss = criterion(out, target)
            pred_choice = out.data.max(1)[1]

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
        loss.backward()
        optimizer.step()

    train_instance_acc = np.mean(mean_correct)
    print('Train Instance Accuracy: %f' % train_instance_acc)


def train_score(net, epoch, optimizer, training_loader, loss_function, troj_dict):
    train_acc = 0
    train_loss = 0
    net.train()
    loss = None
    for batch_index, (data, labels, ids) in enumerate(training_loader):
        labels = labels.type(torch.LongTensor)
        labels = labels.cuda()
        data = data.type(torch.FloatTensor)
        data = data.cuda()
        optimizer.zero_grad()
        outputs = net(data)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        _, pred = outputs.max(1)

        num_correct = (pred == labels).sum()
        acc = int(num_correct) / data.shape[0]
        train_acc += acc
        train_loss += loss.item()
    
    train_acc /= len(training_loader)
    train_loss /= len(training_loader)
    if epoch % 5 == 0:
        print(f'epoch:{epoch}, train_acc:{train_acc}, train loss:{train_loss}')


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-dataset', type=str, default='ModelNet40')
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-point_model', type=str, default='models.pointnet_cls')
    parser.add_argument('-lr', type=int, default=0.05)
    parser.add_argument('-optimizer', type=str, default='Adam')
    parser.add_argument('-num_point', default=3000)
    parser.add_argument('-use_normals', default=False)
    parser.add_argument('-num_category', default=40)
    parser.add_argument('-num_class', default=40)
    args = parser.parse_args()
    return args

# args = get_arg()
# net = train_troj_point(args, 234)
# eval_pointcloud(net, args)


# a = open('./dataset/ModelNet40/bed/train/bed_0290.txt', 'w')




