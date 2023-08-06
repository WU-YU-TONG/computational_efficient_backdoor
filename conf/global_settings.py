import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#directory to save weights file
CHECKPOINT_PATH = './checkpoint/{dataset}/'

DATA_PATH = './dataset/{dataset}'
CLEAN_DATA_PATH = os.path.join(DATA_PATH, 'train_data.npy')
CLEAN_LABELS_PATH = os.path.join(DATA_PATH, 'train_labels.npy')
TROJ_DATA_PATH = os.path.join(DATA_PATH, 'troj_data.npy')
TROJ_LABELS_PATH = os.path.join(DATA_PATH, 'troj_labels.npy')
SCORE_DATA_PATH = os.path.join(DATA_PATH, 'score_data.npy')
SCORE_LABELS_PATH = os.path.join(DATA_PATH, 'score_labels.npy')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test_data.npy')
TEST_LABELS_PATH = os.path.join(DATA_PATH, 'test_labels.npy')
TEST_TROJ_DATA_PATH = os.path.join(DATA_PATH, 'test_troj_data.npy')
TEST_TROJ_LABELS_PATH = os.path.join(DATA_PATH, 'test_troj_labels.npy')

#dataset config
DATASET_CFG = {
    'CIFAR10':{'model': 'image_model', 'num_cls': 10, 'input_dim': (32, 32, 3)},
    'imagenet-10':{'model': 'image_model', 'num_cls': 10, 'input_dim': (224, 224, 3)},
    'gtsrb':{'model': 'image_model', 'num_cls': 43, 'input_dim': (32, 32, 3)},
    'svhn':{'model': 'image_model', 'num_cls': 10, 'input_dim': (32, 32, 3)},
}










