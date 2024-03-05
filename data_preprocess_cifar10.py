
''' TO RUN THIS FILE, FIRSTLY EXTRACT THE FILE 'CIFAR-10-PYTHON.TAR.GZ'
'''

import numpy as np
import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def CreatData_Cifar10():
    x = []
    y = []
    for i in range(1, 6):
        batch_patch = 'cifar-10-batches-py\data_batch_%d' % (i)
        batch_dict = unpickle(batch_patch)
        train_batch=batch_dict[b'data'].astype('float')
        train_labels = np.array(batch_dict[b'labels'])
        x.append(train_batch)
        y.append(train_labels)

    train_data = np.concatenate(x)
    train_labels = np.concatenate(y)

    testpath = os.path.join('cifar-10-batches-py', 'test_batch')
    test_dict = unpickle(testpath)
    test_data = test_dict[b'data'].astype('float')
    test_labels = np.array(test_dict[b'labels'])

    return train_data, train_labels, test_data, test_labels

def CreatData_Cifar100():
    file = '../shadow_dataset/CIFAR100/cifar-100-python/train'
    data = unpickle(file)
    print(data.keys())
    train_data = data[b'data'].astype('float')
    train_labels = data[b'fine_labels']
    file = '../shadow_dataset/CIFAR100/cifar-100-python/test'
    data = unpickle(file)
    print(data.keys())
    test_data = data[b'data'].astype('float')
    test_labels = data[b'fine_labels']
    return train_data, train_labels, test_data, test_labels

def CreatData_Texas100():
    file = '../shadow_dataset/texas100/texas100.npz'
    dataset = np.load(file)
    data = dataset['features'][:50000]
    labels = np.argmax(dataset['labels'][:50000], axis=1)
    test_data = dataset['features'][50000:-1]
    test_labels = np.argmax(dataset['labels'][50000:-1], axis=1)
    print(test_labels.shape)
    return data, labels, test_data, test_labels

def CreatData_Purchase100():
    file = '../shadow_dataset/purchase100/purchase100.npz'
    dataset = np.load(file)
    data = dataset['features'][:150000]
    labels = np.argmax(dataset['labels'][:150000], axis=1)
    test_data = dataset['features'][150000:-1]
    test_labels = np.argmax(dataset['labels'][150000:-1], axis=1)
    print(test_labels.shape)
    return data, labels, test_data, test_labels


def CreatData_gtsrb():
    train_file = '../shadow_dataset/gtsrb/train.npz'
    test_file = '../shadow_dataset/gtsrb/test.npz'
    dataset = np.load(train_file)
    data = dataset['x']
    labels = dataset['y']
    test_dataset = np.load(test_file)
    test_data = test_dataset['x']
    test_labels = test_dataset['y']
    return data, labels, test_data, test_labels

def CreatData_stl10():
    train_file = '../shadow_dataset/stl10/train.npz'
    test_file = '../shadow_dataset/stl10/test.npz'
    dataset = np.load(train_file)
    data = dataset['x']
    labels = dataset['y']
    test_dataset = np.load(test_file)
    test_data = test_dataset['x']
    test_labels = test_dataset['y']
    print(data.shape)
    print(test_data.shape)
    return data, labels, test_data, test_labels

def CreatData_svhn():
    train_file = '../shadow_dataset/svhn/train.npz'
    test_file = '../shadow_dataset/svhn/test.npz'
    dataset = np.load(train_file)
    data = dataset['x']
    labels = dataset['y']
    test_dataset = np.load(test_file)
    test_data = test_dataset['x']
    test_labels = test_dataset['y']
    print(data.shape)
    print(test_data.shape)
    return data, labels, test_data, test_labels


train_data, train_labels, test_data, test_labels = CreatData_svhn()
# np.save('../shadow_dataset/svhn/train_data.npy', train_data)
# np.save('../shadow_dataset/svhn/train_labels.npy', train_labels)
# np.save('../shadow_dataset/svhn/test_data.npy', test_data)
# np.save('../shadow_dataset/svhn/test_labels.npy', test_labels)
data = np.load('../shadow_dataset/gtsrb/train_data.npy')

import matplotlib.pyplot as plt
for num in range(50):
    plt.imshow(data[num])
    plt.savefig(f'./pp{num}.jpg')