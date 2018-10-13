'''
A script to train neural networks.
'''

import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from MNISTModel import MNISTCNN
from MNISTModel import MNISTMLP
from MNISTModel import MNISTLR

'''
from CIFARModel import CIFARMLP
from CIFARModel import LargeNet
from CIFARModel import SmallNet
'''

from utils import trainSaveModel

import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('modelname')
parser.add_argument('dataset')
parser.add_argument('-augmentation', type = int, default = 0)
parser.add_argument('-batch', type = int, default = 128)
parser.add_argument('-epochs', type = int, default = 1)
parser.add_argument('-lr', type = float, default = 0.01)
parser.add_argument('-decay', type = float, default = 1e-6)
parser.add_argument('-momentum', type = float, default = 0.9)
parser.add_argument('-nesterov', type = int, default = 0)

args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Train %s on %s' % (args.modelname, device))

batch_size = args.batch

if args.augmentation == 0:
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
elif args.augmentation == 1:
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding = 4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
else:
    assert(0)

if args.dataset == 'MNIST':
    trainingset = torchvision.datasets.MNIST(root = './Data', train = True, download = True, transform = transform_train)
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size = batch_size, shuffle = True, num_workers = 8)
    testset = torchvision.datasets.MNIST(root = './Data', train = False, download = True, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 8)
elif args.dataset == 'CIFAR':
    trainingset = torchvision.datasets.CIFAR10(root = './Data', train = True, download = True, transform = transform_train)
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size = batch_size, shuffle = True, num_workers = 8)
    testset = torchvision.datasets.CIFAR10(root = './Data', train = False, download = True, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 8)
else:
    assert(0)

if args.modelname == 'MNISTCNN':
    model = MNISTCNN()
    filename = './Model/MNISTCNN.pt'

elif args.modelname == 'MNISTMLP':
    model = MNISTMLP()
    filename = './Model/MNISTMLP.pt'

elif args.modelname == 'MNISTLR':
    model = MNISTLR()
    filename = './Model/MNISTLR.pt'

elif args.modelname == 'CIFACNN':
    model = CIFARCNN()
    filename = './Model/CIFARCNN.pt'
    


optimizer = optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.decay, momentum = args.momentum, nesterov = bool(args.nesterov))
trainSaveModel(filename, model, device, trainloader, testloader, loss_fn = F.cross_entropy, optimizer = optimizer, epochs = args.epochs)



