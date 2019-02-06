'''
Define architectures of neural network on MNIST and CIFAR.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def orthogonal_constraint(model, device, beta):
    ret = 0
    for name, p in model.named_parameters():
        # fully connected layer
        if 'fc' in name and 'weight' in name:
            weight = p
        # convolutional layer
        elif 'conv' in name and 'weight' in name:
            weight = p.view(p.shape[0], -1)
        ret += beta * (torch.mm(weight, weight.t()) - torch.eye(weight.shape[0], dtype = torch.float, device = device)).pow(2).sum()
    return ret

'''
MNIST Models
'''
class MNISTLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        return x

class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = None
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = x.view((-1, 784))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.fc1 = nn.Linear(12 * 12 * 32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = x.view(-1, 12 * 12 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''
CIFAR Models
'''
class CIFARLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        x = x.view((-1, 3 * 32 * 32))
        x = self.fc(x)
        return x

class CIFARMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = x.view((-1, 3 * 32 * 32))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CIFARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(5 * 5 * 128, 256)
        self.drop1 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(256, 256)
        self.drop2 = nn.Dropout(p = 0.5)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, (2, 2))
        x = x.view(-1, 5 * 5 * 128)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x
    
