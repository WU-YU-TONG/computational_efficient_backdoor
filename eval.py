import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

from conf import settings
from util import test_transfer
import provider
from dataset import CIFAR100Train, CIFAR100TrojTrain, Imagenet10TrojTrain
from util import get_network_assmb, get_point_dataset, test_imagenet_transfer


def eval(net, args):
    valid_acc = 0
    troj_asr = 0
    net.eval()
    if args.dataset == 'CIFAR10':
        data = np.load(settings.TEST_DATA_PATH.format(dataset=args.dataset))
        labels = np.load(settings.TEST_LABELS_PATH.format(dataset=args.dataset))
        valid_set = CIFAR100Train(data, labels, test_transfer)
    else:
        valid_set = Imagenet10TrojTrain(if_troj=False, split='test', transform=test_imagenet_transfer, require_index=False)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)
    for _, (data, labels) in enumerate(valid_loader):
        labels = labels.type(torch.LongTensor)
        labels = labels.cuda()
        data = data.type(torch.FloatTensor)
        data = data.cuda()
        outputs = net(data)
        _, pred = outputs.max(1)
        num_correct = (pred == labels).sum()
        acc = int(num_correct) / data.shape[0]
        valid_acc += acc
    valid_acc /= len(valid_loader)

    if args.dataset == 'CIFAR10':
        troj_data = np.load(settings.TEST_TROJ_DATA_PATH.format(dataset=args.dataset))
        troj_labels = np.load(settings.TEST_TROJ_LABELS_PATH.format(dataset=args.dataset))
        troj_set = CIFAR100TrojTrain(troj_data, troj_labels, test_transfer)
    else:
        troj_set = Imagenet10TrojTrain(if_troj=True, split='test', transform=test_imagenet_transfer, require_index=False)
    troj_loader = DataLoader(troj_set, batch_size=32, shuffle=False)
    for _, (data, labels) in enumerate(troj_loader):
        labels = labels.type(torch.LongTensor)
        labels = labels.cuda()
        data = data.type(torch.FloatTensor)
        data = data.cuda()
        outputs = net(data)
        _, pred = outputs.max(1)
        num_correct = (pred == labels).sum()
        acc = int(num_correct) / data.shape[0]
        troj_asr += acc
    troj_asr /= len(troj_loader)

    print(f'valid_acc:{valid_acc}; asr:{troj_asr}')
    return valid_acc, troj_asr

def eval_troj(net, args):
    valid_acc, troj_asr = eval(net, args)
    return valid_acc, troj_asr

def eval_pointcloud(net, args):
    dataset = get_point_dataset(args, 'test')
    test_loader = DataLoader(dataset, 64, shuffle=False)
    net.eval()
    mean_correct = []
    for _, (points, target) in enumerate(test_loader):
        if args.point_model != 'models.pointcnn':
            points = points.data.numpy()
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            pred, trans_feat = net(points)
            pred_choice = pred.data.max(1)[1]
        else:
            rotated_data = provider.rotate_point_cloud(points)
            jittered_data = provider.jitter_point_cloud(rotated_data)  # P_Sampled
            P_sampled = jittered_data
            P_sampled = torch.from_numpy(P_sampled).float()
            P_sampled = Variable(P_sampled, requires_grad=False).cuda()
            label = target.cuda()
            label = Variable(label, requires_grad=False).cuda()
            out = net((P_sampled, P_sampled))
            pred_choice = out.data.max(1)[1]

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
    test_instance_acc = np.mean(mean_correct)
    print('Test Instance Accuracy: %f' % test_instance_acc)

    dataset = get_point_dataset(args, 'test', if_troj=True)
    test_loader = DataLoader(dataset, 64, shuffle=False)
    net.eval()
    mean_correct = []
    for _, (points, target) in enumerate(test_loader):
        if args.point_model != 'models.pointcnn':
            points = points.data.numpy()
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            pred, trans_feat = net(points)
            pred_choice = pred.data.max(1)[1]
        else:
            rotated_data = provider.rotate_point_cloud(points)
            jittered_data = provider.jitter_point_cloud(rotated_data)  # P_Sampled
            P_sampled = jittered_data
            P_sampled = torch.from_numpy(P_sampled).float()
            P_sampled = Variable(P_sampled, requires_grad=False).cuda()
            label = target.cuda()
            label = Variable(label, requires_grad=False).cuda()
            out = net((P_sampled, P_sampled))
            pred_choice = out.data.max(1)[1]

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
    backdoor_instance_acc = np.mean(mean_correct)
    print('Backdoor Instance Accuracy: %f' % backdoor_instance_acc)
    return test_instance_acc, backdoor_instance_acc


