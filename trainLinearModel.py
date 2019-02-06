'''
A simple experiment for linear classifiers' robustness during training, to verify the implementation of Lipchitz constant estimation.
'''

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataset as Dataset

from utils import *

import argparse
import os

from model import MNISTLR
from model import CIFARLR

from estimateLipschitzBound import estimateLipschitzBound

def calculateLinearDistance(model, dataset, q):
    '''
    Take a linear model as input. Output the exact distance to decision boundary.
    q: We are computing Lp distance. q is the dual of p.
    '''

    W, b = list(model.parameters())
    W = W.detach().numpy()
    W = W
    b = b.detach().numpy().reshape((10, 1))
   
    dist = []

    for i in range(len(dataset)):
        img, label = dataset[i]
        x0 = img.reshape((-1, 1))
    
        dist.append(distance(W, b, x0, q = q))

    return np.mean(dist)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('modelname')
    parser.add_argument('dataset')
    parser.add_argument('-p', type = int, default = 1)

    args = parser.parse_args()

    model = modelname2model(args.modelname)

    trainingset, testset = makeDataset(args.dataset)
    trainloader, testloader = makeLoader((trainingset, testset), batch_size = 32)

    p = args.p if args.p < 1e10 else np.inf
    q = dual(p)

    print('p = %f, q = %f' % (p, q))

    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, nesterov = True)

    train(model, torch.device('cpu'), trainloader, testloader, F.cross_entropy, optimizer, epochs = 1, verbose = 2)
     
    average_dist = calculateLinearDistance(model, [testset[i] for i in range(10)], q = q)
    print(average_dist)

    '''
    Be cautious about setting the radius parameter R. It should be large enough so that the distances for linear classifiers won't be truncated.
    '''
    dist_list, _, _ = estimateLipschitzBound(model, torch.device('cpu'), [testset[i] for i in range(10)], Nb = 10, Ns = 10, p = p, q = q, R = 100)
    print(np.mean(dist_list))

if __name__ == '__main__':
    main()

