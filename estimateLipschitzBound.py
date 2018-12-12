'''
Estimate local lipschitz constant by sampling
'''

import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from MNISTModel import *
from CIFARModel import *

from utils import *

import argparse
import os

def sampling(x0, R, Ns, p):
    '''
    Sampling Ns points uniformly in the ball B_p(x0, R)
    Input: x0 is a tensor
    Output: x is a tensor
    '''

    # Step one: sampling gaussian data
    x = np.random.standard_normal((Ns, x0.numel()))
    # Step two: normalizing gaussian data
    x = x / np.linalg.norm(x, ord = p, axis = 1, keepdims = True) * R
    # Step three: move each points by x0
    x = x + x0.reshape((1, -1))

    return torch.tensor(x.reshape((Ns, x0.shape[1], x0.shape[2], x0.shape[3])), dtype = torch.float)

def maximum_grad_norm(model, device, x0, c, j, Nb, Ns, p, q, R):
    '''
    Return the maximum gradient norm in a small ball
    '''
    
    ret = -np.inf
    
    for i in range(Nb):
        
        model.zero_grad()
        
        x = sampling(x0, R, Ns, p = p)
        assert(x.shape == (Ns, 1, 28, 28) or x.shape == (Ns, 3, 32, 32))
        x = torch.tensor(x, device = device, requires_grad = True)

        output = model(x)
        
        g = torch.sum(output[:, c] - output[:, j])
        g.backward()
        
        with torch.no_grad():
            # grad = x.grad.reshape((Ns, -1))
            grad = x.grad.view((Ns, -1))

            assert(grad.shape[1] == 28 * 28 or grad.shape[1] == 3 * 32 * 32)

            grad_norm = torch.norm(grad, p = q, dim = 1)
            assert(grad_norm.shape == (Ns, ))
            
            temp = torch.max(grad_norm)
            ret = max(ret, temp)

    return ret

def targeted_score(model, device, x0, c, j, delta_f, Nb, Ns, p, q, R):
    
    grad_norm = maximum_grad_norm(model = model, device = device, x0 = x0, c = c, j = j, Nb = Nb, Ns = Ns, p = p, q = q, R = R)

    return min(R, (delta_f / grad_norm).item())

def untargeted_score(model, device, x0, c, output, Nb, Ns, p, q, R):
    
    ret = np.inf 

    for j in range(10):
        
        if j == c:
            continue
            
        temp = targeted_score(model = model, device = device, x0 = x0, c = c, j = j, delta_f = output[0, c] - output[0, j], Nb = Nb, Ns = Ns, p = p, q = q, R = R)
        ret = min(ret, temp)

    return ret

def estimateLipschitzBound(model, device, dataset, Nb, Ns, p, q, R):
    '''
    To estimate Lipschitz lower bound by sampling.
    '''

    dist = []
    #target = []
    #prediction = []

    for i in range(len(dataset)):
        img, label = dataset[i]
        img, label = img.to(device), label.to(device)
        
        # Reshape the imput to a tensor with batch_size = 1
        x0 = img.view((1, *tuple(img.shape)))
        
        with torch.no_grad():
            output = model(x0)
            # don't know if it will decrease efficiency putting the following statement outside torch.no_grad()
            c = torch.argmax(output, dim = 1).item()

        # target.append(label)
        # prediction.append(c.item())

        dist_x0 = untargeted_score(model, device, x0, c, output, Nb = Nb, Ns = Ns, p = p, q = q, R = R)
        dist.append(dist_x0)

    return dist

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('modelname')
    parser.add_argument('path')
    parser.add_argument('dataset')
    parser.add_argument('-Nb', type = int, default = 50)
    parser.add_argument('-Ns', type = int, default = 1024)
    parser.add_argument('-p', type = int, default = 2)
    parser.add_argument('-R', type = int, default = 5)

    args = parser.parse_args()
    #print('Estimating distance for %s' % (args.modelname))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    _, testset = makeDataset(args.dataset)

    model = modelname2model(args.modelname)
    model.load_state_dict(torch.load(args.path))
    model.to(device).eval()

    p = args.p if args.p < 1e10 else np.inf
    q = dual(p)

    '''
    if args.random == 1:
        index = np.random.choice(10000, n_samples)
    elif args.random == 0:
        index = np.arange(10000)
    elif args.random == 2:
        index = np.genfromtxt('index.csv', dtype = np.int32)
    else:
        assert(0)
    '''

    printArguments(args)

    print('Estimate %s on %s' % (args.modelname, device))

    dist = estimateLipschitzBound(model, torch.device('cpu'), [testset[i] for i in range(1)], Nb = args.Nb, Ns = args.Ns, p = p, q = q, R = args.R)
    
    '''
    np.savetxt('./output/' + args.modelname + '_estimateddistance.csv', dist, fmt = '%f', delimiter = ',')
    np.savetxt('./output/' + args.modelname + '_index.csv', index, fmt = '%d', delimiter = ',')
    np.savetxt('./output/' + args.modelname + '_target.csv', target, fmt = '%d', delimiter = ',')
    np.savetxt('./output/' + args.modelname + '_prediction.csv', prediction, fmt = '%d', delimiter = ',')
    '''

    print(np.mean(dist))

if __name__ == '__main__':
    main()
