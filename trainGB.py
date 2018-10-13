'''
Train a good classifier and a bad classifier on CIFAR for comparison
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

from CIFARModel import CIFARMLP
from CIFARModel import LargeNet
from CIFARModel import SmallNet

from utils import train, acc

import os.path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 128

transform_train = transforms.Compose([transforms.RandomCrop(32, padding = 4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])

trainingset = torchvision.datasets.CIFAR10(root = './Data', train = True, download = True, transform = transform_test)
trainloader = torch.utils.data.DataLoader(trainingset, batch_size = batch_size, shuffle = True, num_workers = 8)

testset = torchvision.datasets.CIFAR10(root = './Data', train = False, download = True, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 8)

'''
largeNet = LargeNet()
largeNet.to(device)

if os.path.isfile('./Model/largeNet.pt') == True:	
    largeNet.load_state_dict(torch.load('./Model/largeNet.pt'))

optimizer = optim.SGD(largeNet.parameters(), lr = 0.01, weight_decay = 1e-6, momentum = 0.9, nesterov = True)
train(largeNet.train(), device, trainloader, testloader, loss_fn = F.cross_entropy, optimizer = optimizer,  epochs = 200)

largeNet.eval()
print('Large Net Performance : Training Acc : %f, Test Acc : %f' % (acc(largeNet, device, trainloader), acc(largeNet, device, testloader)))

largeNet.to(torch.device('cpu'))
torch.save(largeNet.state_dict(), './Model/largeNet.pt')
'''

cifarMLP = CIFARMLP()
cifarMLP.to(device)

if os.path.isfile('./Model/cifarMLP.pt') == True:	
    cifarMLPNet.load_state_dict(torch.load('./Model/cifarMLP.pt'))

optimizer = optim.SGD(cifarMLP.parameters(), lr = 0.01, weight_decay = 1e-6, momentum = 0.9, nesterov = True)
train(cifarMLP.train(), device, trainloader, testloader, loss_fn = F.cross_entropy, optimizer = optimizer,  epochs = 50)

cifarMLP.eval()
print('Large Net Performance : Training Acc : %f, Test Acc : %f' % (acc(cifarMLP, device, trainloader), acc(cifarMLP, device, testloader)))

cifarMLP.to(torch.device('cpu'))
torch.save(cifarMLP.state_dict(), './Model/cifarMLP.pt')

'''
smallNet = SmallNet()
smallNet.to(device)

if os.path.isfile('./Model/smallNet.pt') == True:	
    smallNet.load_state_dict(torch.load('./Model/smallNet.pt'))

optimizer = optim.SGD(smallNet.parameters(), lr = 0.01, weight_decay = 1e-6, momentum = 0.9, nesterov = True)
train(smallNet.train(), device, trainloader, testloader, loss_fn = F.cross_entropy, optimizer = optimizer, epochs = 20)

print('Small Net Performance : Training Acc : %f, Test Acc : %f' % (acc(smallNet, device, trainloader), acc(smallNet, device, testloader)))

smallNet.to(torch.device('cpu'))
torch.save(smallNet.state_dict(), './Model/smallNet.pt')
'''

