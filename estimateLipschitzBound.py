'''
Estimate local lipschitz constant by sampling
'''

import sys

import numpy as np

import torch

from MNISTModel import *
from CIFARModel import *

from utils import *

import argparse
import os

import pickle

import time

def l2_samples(device, x0, R, Ns, p):
    '''
    Generate samples in a unit ball around origin.
    '''

    '''    
    # Previous sampling method using NumPy.
    # Sampling on CPU then move data to GPU is time consuming.
    # Step one: generate uniform distribution on sphere
    x = np.random.standard_normal((Ns, x0.numel()))
    x = x / np.linalg.norm(x, ord = p, axis = 1, keepdims = True)
    # Step two: generate scaling factor
    z = np.random.uniform(low = 0.0, high = 1.0, size = (Ns, 1))
    z = z ** (1 / x0.numel())
    # Step three: move each points by x0
    x = R * z * x
    x = x.reshape((Ns, x0.shape[1], x0.shape[2], x0.shape[3]))
    '''

    # Update: same sampling method using PyTorch

    # Step one: generate uniform distribution on sphere
    x = torch.randn((Ns, x0.numel()), device = device, dtype = torch.float)
    x = x / torch.norm(x, p = 2, dim = 1, keepdim = True)

    # Step two: generate scaling factor
    z = torch.rand((Ns, 1), device = device, dtype = torch.float)
    z = torch.pow(z, 1 / x0.numel())
    x = R * z * x

    # Step three: reshape samples
    x = x.reshape((Ns, x0.shape[1], x0.shape[2], x0.shape[3]))

    return x

def sampling(device, x0, R, Ns, p):
    '''
    Sampling Ns points uniformly in the ball B_p(x0, R)
    Input: x0 is a tensor
    Output: x is a tensor
    
    Note: one MUST initilize sampling.SAMPLED = False before estimation.
    '''

    '''
    TODO: Add sampling methods for other Lp norm.
    '''
    
    if p == 2:        
        samples = l2_samples(device, x0, R, Ns, p)
    else:
        assert(0)
    
    return torch.tensor(x0 + samples, device = device, dtype = torch.float, requires_grad = True)

def maximum_grad_norm(model, device, x0, c, j, Nb, Ns, p, q, R):
    '''
    Return the maximum gradient norm in a small ball
    '''
    
    ret = -np.inf
    
    for i in range(Nb):
        
        model.zero_grad()
        
        x = sampling(device = device, x0 = x0, R = R, Ns = Ns, p = p)
        assert(x.shape == (Ns, 1, 28, 28) or x.shape == (Ns, 3, 32, 32))

        output = model(x)
        
        g = torch.sum(output[:, c] - output[:, j])
        g.backward()
        
        with torch.no_grad():
            grad = x.grad.view((Ns, -1))
            assert(grad.shape[1] == 28 * 28 or grad.shape[1] == 3 * 32 * 32)

            grad_norm = torch.norm(grad, p = q, dim = 1)
            assert(grad_norm.shape == (Ns, ))
            
            temp = torch.max(grad_norm)
            
            ret = max(ret, temp)
    
    return ret

def targeted_score(model, device, x0, c, j, delta_f, Nb, Ns, p, q, R):
    
    Lc = maximum_grad_norm(model = model, device = device, x0 = x0, c = c, j = j, Nb = Nb, Ns = Ns, p = p, q = q, R = R)

    return min(R, delta_f / Lc)

def untargeted_score(model, device, x0, c, output, Nb, Ns, p, q, R):
    
    ret = np.inf 

    for j in range(10):
        
        if j == c:
            continue
            
        temp = targeted_score(model = model, device = device, x0 = x0, c = c, j = j, delta_f = (output[0, c] - output[0, j]).item(), Nb = Nb, Ns = Ns, p = p, q = q, R = R)
        ret = min(ret, temp)

    return ret

def estimateLipschitzBound(model, device, dataset, Nb, Ns, p, q, R):
    '''
    To estimate Lipschitz lower bound by sampling.
    '''

    model.to(device).eval()

    print('Estimate %s on %s' % (model.__class__.__name__, device))

    dist = []
    target = []
    prediction = []
    
    for i in range(len(dataset)):
        '''
        On MNIST, label is a torch tensor with only one element; while on CIFAR, label is an int.
        Convert it to long tensor explicitly.
        '''
        img, label = dataset[i]
        img, label = img.to(device), torch.tensor(label, device = device, dtype = torch.long)

        # Reshape the imput to a tensor with batch_size = 1
        x0 = img.view((1, *tuple(img.shape)))

        with torch.no_grad():
            output = model(x0)
            c = torch.argmax(output, dim = 1).item()
        
        target.append(label.item())
        prediction.append(c)
        
        dist_x0 = untargeted_score(model, device, x0, c, output, Nb = Nb, Ns = Ns, p = p, q = q, R = R)
        dist.append(dist_x0)

    return dist, target, prediction

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('modelname')
    parser.add_argument('model_path')
    parser.add_argument('store_path')
    parser.add_argument('dataset')
    parser.add_argument('-train', type = int, default = 0, help = 'Evaluate on training set or test set.')
    parser.add_argument('-n', type = int, default = 500, help = 'Number of samples that need to estimate.')
    parser.add_argument('-Nb', type = int, default = 50)
    parser.add_argument('-Ns', type = int, default = 1024)
    parser.add_argument('-p', type = int, default = 2)
    parser.add_argument('-R', type = int, default = 5)
    parser.add_argument('-ckpt', type = int, default = 0, help = 'Indicate whether model path is a check point or not.')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the pretrained model.
    model = modelname2model(args.modelname)
    
    if bool(args.ckpt) == False:
        model.load_state_dict(torch.load(args.model_path))
    else:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    p = args.p if args.p < 1e10 else np.inf
    q = dual(p)
    
    # Prepare the subset that need to estimate. Fix the random seed to make sure the subset is the same every time.
    trainingset, testset = makeDataset(args.dataset)
    
    np.random.seed(0)
    
    if bool(args.train) == True:
        index = np.random.random_choice(len(trainingset), size = args.n, replace = False)
        subset = [trainingset[i] for i in index]
    else:
        index = np.random.choice(len(testset), size = args.n, replace = False)
        subset = [testset[i] for i in index]

    # Reset the random seed.
    np.random.seed()

    printArguments(args)
    
    dist, target, prediction  = estimateLipschitzBound(model, device, subset, Nb = args.Nb, Ns = args.Ns, p = p, q = q, R = args.R)
    
    estimation = {'index' : index, \
                  'target' : target, \
                  'prediction' : prediction, \
                  'dist' : dist, \
                  'config' : args, \
                  }
    
    print(np.mean(dist))
    
    '''
    with open(args.store_path, 'wb+') as f:
        pickle.dump(estimation, f)
    '''
    torch.save(estimation, args.store_path)

if __name__ == '__main__':
    main()

