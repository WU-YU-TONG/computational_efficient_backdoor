import os
import argparse
import importlib
import torch
import numpy as np
import random as rd

from backdoor_score import GS, grad_scoring, RD_scoring, RD_point
from util import compose_troj_data, compose_troj_point, get_network_assmb
from training import train_troj_model, forgetting_score, train_troj_point, forgetting_point
from eval import eval_troj, eval_pointcloud

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-diet_mode', type=str, default='RD_SCORE')
    parser.add_argument('-task', type=str, default='image')
    parser.add_argument('-repeat', type=int, default=30)
    parser.add_argument('-epoch', type=int, default=40)
    parser.add_argument('-times', type=int, default=30)
    parser.add_argument('-use_troj', action='store_true')
    parser.add_argument('-score_epoch', type=int, default=6)
    parser.add_argument('-milestone', type=list, default=[25, 35, 45])
    parser.add_argument('-target', type=int, default=1)
    parser.add_argument('-perc', type=float, default=0.0011)
    parser.add_argument('-ratio', type=float, default=0.5)
    parser.add_argument('-decay', type=float. default=0.93)
    parser.add_argument('-dataset', type=str, default='CIFAR10')
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-score_batch_size', type=int, default=128)
    parser.add_argument('-optimizer', type=str, default='SGD')
    parser.add_argument('-score_optimizer', type=str, default='SGD')
    parser.add_argument('-model_name', type=str, default='resnet18')
    parser.add_argument('-score_model_name', type=str, default='resnet18')
    parser.add_argument('-lr', type=int, default=0.05)
    parser.add_argument('-score_lr', type=int, default=0.05)
    parser.add_argument('-point_model', type=str, default='models.pointcnn')
    parser.add_argument('-score_point_model', type=str, default='models.pointnet_cls')

    parser.add_argument('-cons_path', type=str, default='./cons.txt')

    parser.add_argument('-num_point', type=int, default=2048)
    parser.add_argument('-use_normals', default=False)
    parser.add_argument('-num_category', default=40)
    parser.add_argument('-num_class', default=40)
    
    parser.add_argument('-gpu', type=bool, default=True)
    parser.add_argument('-cuda', type=str, default='0')
    parser.add_argument('-random', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args.score_point_model = args.point_model
    times = 10
    seeds = list(range(args.repeat))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    lasr, lacc = [], []
    lasrr, laccr = [], []
    if args.task == 'image':
        troj_list, out_list, lenth = compose_troj_data(args)
    else:
        troj_list, out_list, lenth = compose_troj_point(args)

    if os.path.exists(args.cons_path):
        f = open(args.cons_path, 'a')
    else:
        f = open(args.cons_path, 'w')
    f.writelines(f'==============={args.task}===============\n')
    f.close()
    if args.task == 'image':
        for seed in seeds:

            if args.random:
                troj_list, out_list, lenth = compose_troj_data(args)
                model = get_network_assmb(args.model_name, args)
                train_troj_model(model, args, seed, troj_list=troj_list) 
                acc, asr = eval_troj(model, args)
                lasrr.append(asr)
                laccr.append(acc)

                if args.diet_mode == 'forget':
                    troj_list = GS(args, troj_list, out_list, forgetting_score)
                elif args.diet_mode == 'grad':
                    troj_list = GS(args, troj_list, out_list, grad_scoring)
                else:
                    troj_list = GS(args, troj_list, out_list, RD_scoring)
                print(f'{len(troj_list)} final trojed data')
                compose_troj_data(args, troj_list)
                model = get_network_assmb(args.model_name, args)
                train_troj_model(model, args, seed, troj_list=troj_list)
                acc, asr = eval_troj(model, args)
                lasr.append(asr)
                lacc.append(acc)
    else:
        for seed in seeds:
            if args.random:
                print(f'{len(troj_list)} final trojed data')
                model = importlib.import_module(args.point_model)
                net = model.get_model(num_class=args.num_class)
                train_troj_point(net, args, seed, troj_list=troj_list) 
                acc, asr = eval_pointcloud(net, args)
                lasrr.append(asr)
                laccr.append(acc)
            else:
                troj_list, out_list, lenth = compose_troj_point(args)

                if args.diet_mode == 'forget':
                    troj_list = GS(args, troj_list, out_list, forgetting_point)
                elif args.diet_mode == 'grad':
                    troj_list = GS(args, troj_list, out_list, grad_scoring)
                else:
                    troj_list = GS(args, troj_list, out_list, RD_point)
                print(f'{len(troj_list)} final trojed data')
                model = importlib.import_module(args.point_model)
                net = model.get_model()
                train_troj_point(net, args, seed, troj_list=troj_list)
                acc, asr = eval_pointcloud(net, args)
                lasr.append(asr)
                lacc.append(acc)      


    # lasrr = np.array(lasrr)
    # laccr = np.array(laccr)
    # asr_aver = np.average(lasrr)
    # acc_aver = np.average(laccr)
    # asr_std = np.std(lasrr)
    # acc_std = np.std(laccr)
    # if os.path.exists(args.cons_path):
    #     f = open(args.cons_path, 'a')
    # else:
    #     f = open(args.cons_path, 'w')
    #
    # if args.task == 'image':
    #     f.writelines(f'RANDOM: perc:{args.perc}, ratio:{args.ratio}, troj model:{args.model_name}\n')
    # else:
    #     f.writelines(f'RANDOM: perc:{args.perc}, ratio:{args.ratio}, troj model:{args.point_model}\n')
    # f.writelines(f'ASR: {asr_aver}+-{asr_std};\tACC: {acc_aver}+-{acc_std};\n\n')
    # f.close()
    # lasr = np.array(lasr)
    # lacc = np.array(lacc)
    # asr_aver = np.average(lasr)
    # acc_aver = np.average(lacc)
    # asr_std = np.std(lasr)
    # acc_std = np.std(lacc)
    # if os.path.exists(args.cons_path):
    #     f = open(args.cons_path, 'a')
    # else:
    #     f = open(args.cons_path, 'w')
    # f.writelines(f'perc:{args.perc}, ratio:{args.ratio}\n')
    # if args.task == 'image':
    #     f.writelines(f'score model:{args.score_model_name}, troj model:{args.model_name}, score_epoch:{args.score_epoch}\n')
    # else:
    #     f.writelines(f'score model:{args.score_point_model}, troj model:{args.point_model}\n')
    # f.writelines(f'troj batch:{args.batch_size}, score batch:{args.score_batch_size}, troj optimizer:{args.optimizer}, score optimizer:{args.score_optimizer}\n')
    # f.writelines(f'lr:{args.lr}, score lr:{args.score_lr}, mode:{args.diet_mode}, iteration:{args.times}, dataset:{args.dataset}\n')
    # f.writelines(f'ASR: {asr_aver}+-{asr_std};\tACC: {acc_aver}+-{acc_std};\n\n')
    # f.close()


