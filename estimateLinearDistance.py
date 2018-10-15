'''
Compute the distance to boundary fo linear classifiers
'''

import numpy as np

import torch
import os.path
import argparse

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from MNISTModel import MNISTLR

#from CIFARModel import SmallNet


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

parser = argparse.ArgumentParser()
parser.add_argument('modelname')
parser.add_argument('dataset')
parser.add_argument('-n', type=int, default = 100)
parser.add_argument('-p', type=int, default = 2)

args = parser.parse_args()

if args.dataset == 'MNIST':
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root = './Data', train = False, download = True, transform = transform)
elif args.dataset == 'CIFAR':
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root = './Data', train = False, download = True, transform = transform)
else:
    assert(0)

if args.modelname == 'MNISTLR':
    model = MNISTLR()
    model.load_state_dict(torch.load('./Model/MNISTLR.pt'))
    model.eval()

else:
    assert(0)
    model = SmallNet()
    model.load_state_dict(torch.load('./Model/smallNet.pt'))
    model.eval()

W, b = list(model.parameters())
W = W.detach().numpy()
b = b.detach().numpy().reshape((10, 1))

n_samples = args.n

p = args.p if args.p < 10000000000 else np.inf
if p == np.inf:
    q = 1
elif p == 1:
    q = np.inf
else:
    q = p / (p - 1)

index = np.arange(10000)
dist = []

for i in range(n_samples):

    input, label = testset[index[i]]
    
    x0 = input.reshape((-1, 1))

    dist.append(distance(W, b, x0, q = q))

print(np.mean(dist))





