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

from CIFARModel import CIFARCNN
from CIFARModel import CIFARMLP
from CIFARModel import LargeNet
from CIFARModel import SmallNet

import os.path

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('modelname')
parser.add_argument('-n', type = int, default = 500)
parser.add_argument('-Nb', type = int, default = 50)
parser.add_argument('-Ns', type = int, default = 1024)
parser.add_argument('-R', type = int, default = 5)
parser.add_argument('-random', type = int, default = 1)

args = parser.parse_args()
print('Estimating distance for %s' % (args.modelname))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Use device %s' % (device))

def sampling(x0, R, Ns):
    '''
    Sampling Ns points uniformly in the ball B(x0, R)
    Input: x0 is a tensor
    Output: x is a tensor
    '''

    # Step one: sampling gaussian data
    x = np.random.standard_normal((Ns, x0.numel()))
    # Step two: normalizing gaussian data
    x = x / np.linalg.norm(x, ord = 2, axis = 1, keepdims = True) * R
    # Step three: move each points by x0
    x = x + x0.reshape((1, -1))

    return torch.tensor(x.reshape((Ns, x0.shape[1], x0.shape[2], x0.shape[3])), dtype = torch.float)


def maximum_grad_norm(model, x0, R, c, j, Nb, Ns):
    '''
    Return the maximum gradient norm in a small ball
    '''
    
    ret = 0
    
    for i in range(Nb):
        
        model.zero_grad()
        
        x = sampling(x0, R, Ns)
        assert(x.shape == (Ns, 1, 28, 28) or x.shape == (Ns, 3, 32, 32))
        x = torch.tensor(x, device = device, requires_grad = True)

        output = model(x)
        
        g = torch.sum(output[:, c] - output[:, j])
        g.backward()
        
        with torch.no_grad():
            grad = x.grad.reshape((Ns, -1))
            assert(grad.shape[1] == 784 or grad.shape[1] == 3 * 32 * 32)
            grad_norm = torch.norm(grad, p = 2, dim = 1)
            assert(grad_norm.shape == (Ns, ))
            temp = torch.max(grad_norm) 
            ret = max(ret, temp)    

    return ret

def targeted_score(model, x0, R, c, j, delta_f, Nb, Ns):
    
    lipschitz = maximum_grad_norm(model, x0, R, c, j, Nb, Ns)

    return min(R, (delta_f / lipschitz).item())

def untargeted_score(model, x0, output, R, Nb, Ns):
    
    ret = R 

    c = torch.argmax(output).item()
    
    for j in range(10):
        
        if j == c:
            continue
            
        temp = targeted_score(model, x0, R, c, j, output[0, c] - output[0, j], Nb, Ns)
        ret = min(ret, temp)

    return ret
 

if args.modelname == 'MNISTMLP':
    transform = transforms.Compose([transforms.ToTensor()]) 
    testset = torchvision.datasets.MNIST(root = './Data', train = False, download = True, transform = transform)
    
    model = MLP()
    model.load_state_dict(torch.load('./Model/MNISTMLP.pt'))
    model.to(device).eval()
    
elif args.modelname == 'LargeNet':
    transform = transforms.Compose([transforms.ToTensor()]) 
    testset = torchvision.datasets.CIFAR10(root = './Data', train = False, download = True, transform = transform)
    
    model = LargeNet()
    model.load_state_dict(torch.load('./Model/largeNet.pt'))
    model.to(device).eval()

elif args.modelname == 'CIFARMLP':
    transform = transforms.Compose([transforms.ToTensor()]) 
    testset = torchvision.datasets.CIFAR10(root = './Data', train = False, download = True, transform = transform)
    
    model = CIFARMLP()
    model.load_state_dict(torch.load('./Model/cifarMLP.pt'))
    model.to(device).eval()

elif args.modelname == 'SmallNet':
    transform = transforms.Compose([transforms.ToTensor()]) 
    testset = torchvision.datasets.CIFAR10(root = './Data', train = False, download = True, transform = transform)
    
    model = SmallNet()
    model.load_state_dict(torch.load('./Model/smallNet.pt'))
    model.to(device).eval()

else:
    print('Model not found!')
    sys.exit(0)

R = args.R
Nb = args.Nb
Ns = args.Ns
n_samples = args.n

if args.random == 1:
    index = np.random.choice(10000, n_samples)
elif args.random == 0:
    index = np.arange(10000)
else:
    index = np.genfromtxt('Index.csv', dtype = np.int32)

dist = []
target = []
prediction = []

for i in range(n_samples):
    
    input, label = testset[index[i]]
    #label might be a int variable, not a tensor
    #input, label = input.to(device), label.to(device)
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

    dist_x0 = untargeted_score(model = model, x0 = x0, output = output, R = R, Nb = Nb, Ns = Ns)
    dist.append(dist_x0)

np.savetxt('./Output/' + args.modelname + '_EstimatedDistance.csv', dist, fmt = '%f', delimiter = ',')
np.savetxt('./Output/' + args.modelname + '_Index.csv', index, fmt = '%d', delimiter = ',')
np.savetxt('./Output/' + args.modelname + '_Target.csv', target, fmt = '%d', delimiter = ',')
np.savetxt('./Output/' + args.modelname + '_Prediction.csv', prediction, fmt = '%d', delimiter = ',')

print(np.mean(dist))


