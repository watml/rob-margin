'''
Estimate local lipschitz constant by sampling
'''
import sys

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
from CIFARModel import CIFARCNN
from CIFARModel import CIFARMLP
from CIFARModel import LargeNet
from CIFARModel import SmallNet
'''

import os.path

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('modelname')
parser.add_argument('dataset')
parser.add_argument('-n', type = int, default = 500)
parser.add_argument('-Nb', type = int, default = 50)
parser.add_argument('-Ns', type = int, default = 1024)
parser.add_argument('-R', type = int, default = 5)
parser.add_argument('-random', type = int, default = 1)
parser.add_argument('-p', type = int, default = 2)

args = parser.parse_args()
#print('Estimating distance for %s' % (args.modelname))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Use device %s' % (device))

def printArguments(args):

    for arg in vars(args):
        if arg == 'p' or arg == 'q':
            continue
        print('%s = %s' % (arg, getattr(args, arg)), end = ', ')

def sampling(x0, R, Ns, p = 2):
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

def maximum_grad_norm(model, x0, R, c, j, Nb, Ns, p = 2, q = 2):
    '''
    Return the maximum gradient norm in a small ball
    '''
    
    ret = 0
    
    for i in range(Nb):
        
        model.zero_grad()
        
        x = sampling(x0, R, Ns, p = p)
        assert(x.shape == (Ns, 1, 28, 28) or x.shape == (Ns, 3, 32, 32))
        x = torch.tensor(x, device = device, requires_grad = True)

        output = model(x)
        
        g = torch.sum(output[:, c] - output[:, j])
        g.backward()
        
        with torch.no_grad():
            grad = x.grad.reshape((Ns, -1))
            assert(grad.shape[1] == 784 or grad.shape[1] == 3 * 32 * 32)
           
            grad_norm = torch.norm(grad, p = q, dim = 1)
            assert(grad_norm.shape == (Ns, ))
            
            temp = torch.max(grad_norm) 
            ret = max(ret, temp)    

    return ret

def targeted_score(model, x0, R, c, j, delta_f, Nb, Ns, p = 2, q = 2):
    
    lipschitz = maximum_grad_norm(model, x0, R, c, j, Nb, Ns, p = p, q = q)

    return min(R, (delta_f / lipschitz).item())

def untargeted_score(model, x0, output, R, Nb, Ns, p = 2, q = 2):
    
    ret = R 

    c = torch.argmax(output).item()
    
    for j in range(10):
        
        if j == c:
            continue
            
        temp = targeted_score(model, x0, R, c, j, output[0, c] - output[0, j], Nb, Ns, p = p, q = q)
        ret = min(ret, temp)

    return ret

if args.dataset == 'MNIST':
    transform = transforms.Compose([transforms.ToTensor()]) 
    testset = torchvision.datasets.MNIST(root = './Data', train = False, download = True, transform = transform)
elif args.dataset == 'CIFAR':
    transform = transforms.Compose([transforms.ToTensor()]) 
    testset = torchvision.datasets.CIFAR10(root = './Data', train = False, download = True, transform = transform)
else:
    assert(0)

if args.modelname == 'MNISTCNN':
    model = MNISTCNN()
    model.load_state_dict(torch.load('./Model/MNISTCNN.pt'))
    model.to(device).eval()

elif args.modelname == 'MNISTMLP':
    model = MNISTMLP()
    model.load_state_dict(torch.load('./Model/MNISTMLP.pt'))
    model.to(device).eval()

elif args.modelname == 'MNISTLR':
    model = MNISTLR()
    model.load_state_dict(torch.load('./Model/MNISTLR.pt'))
    model.to(device).eval()
    
elif args.modelname == 'LargeNet':
    model = LargeNet()
    model.load_state_dict(torch.load('./Model/largeNet.pt'))
    model.to(device).eval()

elif args.modelname == 'CIFARMLP':
    model = CIFARMLP()
    model.load_state_dict(torch.load('./Model/cifarMLP.pt'))
    model.to(device).eval()

elif args.modelname == 'SmallNet':
    model = SmallNet()
    model.load_state_dict(torch.load('./Model/smallNet.pt'))
    model.to(device).eval()

else:
    assert(0)

R = args.R
Nb = args.Nb
Ns = args.Ns
n_samples = args.n
p = args.p if args.p < 1e10 else np.inf
if p == np.inf:
    q = 1
elif p == 1:
    q = np.inf
else:
    q = p / (p - 1)

if args.random == 1:
    index = np.random.choice(10000, n_samples)
elif args.random == 0:
    index = np.arange(10000)
elif args.random == 2:
    index = np.genfromtxt('Index.csv', dtype = np.int32)
else:
    assert(0)

printArguments(args)
print('p = %s, q = %s' % (p, q))

dist = []
target = []
prediction = []

for i in range(n_samples):
    
    input, label = testset[index[i]]
    input, label = input.to(device), label
    
    x0 = input.reshape((1, *tuple(input.shape)))
    
    with torch.no_grad():
        output = model(x0)
    
    '''
    # Ignore the sample if misclassified
    if torch.argmax(output, dim = 1) != label:
        continue
    '''

    c = torch.argmax(output, dim = 1)
    target.append(label)
    prediction.append(c.item())

    dist_x0 = untargeted_score(model = model, x0 = x0, output = output, R = R, Nb = Nb, Ns = Ns, p = p, q = q)
    dist.append(dist_x0)

np.savetxt('./Output/' + args.modelname + '_EstimatedDistance.csv', dist, fmt = '%f', delimiter = ',')
np.savetxt('./Output/' + args.modelname + '_Index.csv', index, fmt = '%d', delimiter = ',')
np.savetxt('./Output/' + args.modelname + '_Target.csv', target, fmt = '%d', delimiter = ',')
np.savetxt('./Output/' + args.modelname + '_Prediction.csv', prediction, fmt = '%d', delimiter = ',')

print(np.mean(dist))


