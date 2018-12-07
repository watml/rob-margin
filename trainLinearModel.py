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

from estimateLinearDistance import estimateModelDistance

from MNISTModel import MNISTLR
from CIFARModel import CIFARLR

def distance2(w, b, x, q = 2):
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

    c = np.argmax(np.dot(W, x) + b, axis = 0)
    #d = x.shape[0]

    ret = 1e10
    for i in range(0, 10):
        
        if i == c:
            continue
        
        temp = distance2(W[i, :].reshape((-1, 1)) - W[c, :].reshape((-1, 1)), b[i, 0] - b[c, 0], x, q = q) 
        ret = min(ret, temp)
        
    return ret

def estimateModelDistance(model, X, q = 2):
    
    W, b = list(model.parameters())
    W = W.detach().numpy()
    b = b.detach().numpy().reshape((10, 1))
   
    dist = []

    for i in range(len(X)):
        input, label = X[i]
        x0 = input.reshape((-1, 1))
    
        dist.append(distance(W, b, x0, q = q))

    return np.mean(dist)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('modelname')
    parser.add_argument('dataset')
    parser.add_argument('-n', type = int, default = 100)
    parser.add_argument('-p', type = int, default = 2)

    args = parser.parse_args()

    model = modelname2model(args.modelname)

    trainingset, testset = makeDataset(args.dataset)
    trainloader, testloader = makeLoader((trainingset, testset), batch_size = 32)

    n_samples = args.n

    p = args.p if args.p < 1e10 else np.inf
    q = dual(p)

    optimizer = optim.SGD(model.parameters(), lr = 0.01)

    for i in range(20):
        
        total_loss = 0

        for img, label in trainloader:
            
            model.zero_grad()

            output = model(img)
            loss = F.cross_entropy(output, label)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(trainloader)

        #print('Epoch : %d, Loss : %f' % (i + 1, total_loss))
        print('Epoch : %d, Loss : %f, Test Acc : %f, Avg Dist : %f' % (i + 1, total_loss, acc(model, device, testloader), estimateModelDistance(model, testset, q = 2)))

    '''
    TODO: Compare two estimation methods.
    '''
    average_dist = calculateDistance(model, testset, q = q)
    print(average_dist)

if __name__ == '__main__':
    main()
