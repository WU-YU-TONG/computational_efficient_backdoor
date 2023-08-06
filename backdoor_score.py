import os
import random as rd
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import importlib

from conf import settings
from training import train_grad_score, train_score, train_troj_point
from dataset import CIFARDatasetforScore, Imagenet10Score
from util import get_network_assmb, test_transfer, compose_troj_data, test_imagenet_transfer
from data_utils.ModelNetDataLoader import ModelNetDataLoaderScore
from data_utils.ScanObjectDataLoader import ScanObjectDataLoaderScore

def compute_grad(model, loss_func, sample, target):

    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)

    prediction = model(sample)
    loss = loss_func(prediction, target)

    return torch.autograd.grad(loss, list(model.parameters()))


def compute_grad_norm_scores(model, input, ground_truth, loss_func):
    loss = loss_func(model(*input), ground_truth)
    loss.backward()
    loss_grads = []
    for param in model.parameters():
        loss_grads.append(param.grad.clone().view(-1))
        param.grad.data.zero_()
    loss_grads = torch.cat(loss_grads)
    score = torch.linalg.norm(loss_grads).cpu().numpy()
    # print(score)
    return score


def score_samples(model, train_loader, loss_func):
    scores = []
    ids = []
    for i, (data, labels) in enumerate(train_loader):
        labels = labels.type(torch.LongTensor)
        labels = labels.cuda()
        data = data.unsqueeze(0)
        data = data.type(torch.FloatTensor)
        data = data.cuda()
        score = compute_grad_norm_scores(model, data, labels, loss_func)
        scores.append(score)
    return scores, ids


def RD_score_samples(model, train_loader):
    scores = []
    ids = []
    for i, (data, labels) in enumerate(train_loader):
        labels = labels.type(torch.LongTensor)
        labels = labels.cuda()
        data = data.type(torch.FloatTensor)
        data = data.cuda()
        output = model(data)
        # print(output)
        prob = nn.Softmax()(output)
        # print(prob)
        # print(prob.shape)
        for i in range(prob.shape[0]):
            one_hot = torch.zeros_like(output[i])
            one_hot[int(labels[i])] = 1
            score = torch.linalg.norm(prob[i]-one_hot).item()
            scores.append(score)
    return scores, ids


def RD_score_point(model, train_loader):
    scores = []
    ids = []
    for i, (points, target) in enumerate(train_loader):
        points = points.data.numpy()
        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, trans_feat = model(points)
        prob = nn.Softmax()(pred)
        for i in range(prob.shape[0]):
            one_hot = torch.zeros_like(pred[i])
            one_hot[int(target[i])] = 1
            score = torch.linalg.norm(prob[i]-one_hot).item()
            scores.append(score)
    return scores, ids


def GS(args, troj_list, out_list, method):
    for _ in range(args.times):
        
        score = method(args, troj_list)
        print(len(score))
        score_list = np.argsort(np.array(score))
        lens = len(troj_list)
        diet_list = [troj_list[i] for i in score_list[0:int(len(score)*args.ratio + 0.5)]]
        troj_list = [i for i in troj_list if i not in diet_list]
        add_list = rd.sample(out_list, int(lens * args.ratio + 0.5))
        out_list = [i for i in out_list if i not in add_list]
        out_list.extend(diet_list)
        troj_list.extend(add_list)
        if args.task == 'image':
            compose_troj_data(args, troj_list)
    
    return troj_list


def grad_scoring(args, troj_list):
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
    # model set up
    train_grad_score(args)
    data = np.load(settings.SCORE_DATA_PATH.format(dataset=args.dataset))
    labels = np.load(settings.SCORE_LABELS_PATH.format(dataset=args.dataset))

    score_dset = CIFARDatasetforScore(data, labels, troj_list=troj_list, transform=test_transfer)
    print(f'There are {len(score_dset)} troj_data.')
    score_loader =  DataLoader(score_dset, 1, shuffle=False)

    model = get_network_assmb(args.score_model_name, args)
    # model = nn.DataParallel(model).cuda() 
    model_path = os.path.join(settings.CHECKPOINT_PATH.format(dataset=args.dataset), f'{args.score_model_name}_score_model.pth')

    print('Loading saved model from: ' + model_path)
    model.load_state_dict(torch.load(model_path))
    loss_func = nn.CrossEntropyLoss()
    model.train()
    scores, _ = score_samples(model, score_loader, loss_func=loss_func)
    return scores


def RD_scoring(args, troj_list):
    train_grad_score(args, troj_list)
    if args.dataset == 'CIFAR10':
        data = np.load(settings.SCORE_DATA_PATH.format(dataset=args.dataset))
        labels = np.load(settings.SCORE_LABELS_PATH.format(dataset=args.dataset))
        score_dset = CIFARDatasetforScore(data, labels, troj_list=troj_list, transform=test_transfer)
    else:
        score_dset = Imagenet10Score(troj_list=troj_list, transform=test_imagenet_transfer)
    print(f'There are {len(score_dset)} troj_data.')
    score_loader =  DataLoader(score_dset, 1, shuffle=False)

    model = get_network_assmb(args.score_model_name, args)
    # model = nn.DataParallel(model).cuda() 
    model_path = os.path.join(settings.CHECKPOINT_PATH.format(dataset=args.dataset), f'{args.score_model_name}_score_model.pth')

    print('Loading saved model from: ' + model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    scores, _ = RD_score_samples(model, score_loader)
    return scores

def RD_point(args, troj_list):
    model = importlib.import_module(args.score_point_model)
    net = model.get_model(num_class=args.num_class)
    train_troj_point(net, args, troj_list=troj_list, mode='score', seed=None)
    if args.dataset=='ModelNet40':
        score_dset = ModelNetDataLoaderScore(root='./dataset/ModelNet40/', args=args, troj_list=troj_list)
    else:
        score_dset = ScanObjectDataLoaderScore(root='./dataset/ScanObject/', args=args, troj_list=troj_list)
    score_loader = DataLoader(score_dset, 64, shuffle=False)
    scores, _ = RD_score_point(net, score_loader)
    return scores
