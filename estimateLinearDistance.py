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

from CIFARModel import SmallNet


def distance2(w, b, x):
    '''
    w is 3072 * 1
    b is 1 * 1
    x is 3072 * 1
    '''
    return np.linalg.norm(np.dot(w.T, x) + b) / np.linalg.norm(w)
    
def distance(W, b, x):
    '''
    W is 10 * 3072
    b is 10 * 1
    x is 3072 * 1
    
    d is scalar
    '''

    c = np.argmax(np.dot(W, x) + b, axis = 0)
    
    d = 1e10
    for i in range(0, 10):
        
        if i == c:
            continue
        
        temp = distance2(W[i, :].reshape((3072, 1)) - W[c, :].reshape((3072, 1)), b[i, 0] - b[c, 0], x) 
        d = min(d, temp)
        
    return d

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default = 100)

args = parser.parse_args()

transform = transforms.Compose([transforms.ToTensor()])

testset = torchvision.datasets.CIFAR10(root = './Data', train = False, download = True, transform = transform)

model = SmallNet()
model.load_state_dict(torch.load('./Model/smallNet.pt'))

W, b = list(model.parameters())
W = W.detach().numpy()
b = b.detach().numpy().reshape((10, 1))

n_samples = args.n
index = np.arange(10000)
dist = []

for i in range(n_samples):

    input, label = testset[index[i]]
    
    x0 = input.reshape((3072, 1))

    dist.append(distance(W, b, x0))

print(np.mean(dist))
