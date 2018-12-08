'''
A simple experiment for linear classifiers' robustness during training, verifying the implementation of Lipchitz constant estimation.
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

from MNISTModel import MNISTLR
from CIFARModel import CIFARLR

from estimateLipschitzBound import estimateLipschitzBound

def __distance__(w, b, x, q = 2):
    '''
    w is d * 1
    b is 1 * 1
    x is d * 1
    '''
    return np.linalg.norm(np.dot(w.T, x) + b) / np.linalg.norm(w.squeeze(), ord = q)
    
def distance(W, b, x, q = 2):
    '''
    W is 10 * d
    b is 10 * 1
    x is d * 1
    
    ret is scalar
    '''

    d = x.shape[0]

    c = np.argmax(np.dot(W, x) + b, axis = 0)

    ret = np.inf
    for i in range(0, 10):
        
        if i == c:
            continue
        
        temp = __distance__((W[i, :] - W[c, :]).reshape((d, 1)), b[i, 0] - b[c, 0], x, q = q) 
        ret = min(ret, temp)
        
    return ret

def calculateLinearDistance(model, dataset, q = 2):
    '''
    Take a linear model as input. Output the exact distance to decision boundary.
    q: We are computing Lp distance. q is the dual of p.
    '''

    W, b = list(model.parameters())
    W = W.detach().numpy()
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
    #parser.add_argument('-n', type = int, default = 100)
    parser.add_argument('-p', type = int, default = 2)

    args = parser.parse_args()

    model = modelname2model(args.modelname)

    trainingset, testset = makeDataset(args.dataset)
    trainloader, testloader = makeLoader((trainingset, testset), batch_size = 32)

    #n_samples = args.n

    p = args.p if args.p < 1e10 else np.inf
    q = dual(p)

    optimizer = optim.SGD(model.parameters(), lr = 0.005)

    train(model, torch.device('cpu'), trainloader, testloader, F.cross_entropy, optimizer, epochs = 1, verbose = 2)
        
    '''
    TODO: Compare two estimation methods.
    '''
    average_dist = calculateLinearDistance(model, [testset[i] for i in range(500)], q = q)
    print(average_dist)

    dist_list = estimateLipschitzBound(model, torch.device('cpu'), [testset[i] for i in range(500)], Nb = 1, Ns = 10, p = p, q = q, R = 5)
    print(np.mean(dist_list))

if __name__ == '__main__':
    main()
