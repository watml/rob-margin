'''
Define architecture of neural network on MNIST and CIFAR.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
Helper functions returning orthogonal constraint
'''

'''
def orthogonal_constraint_fc(fc, beta):
    weight = fc.weight
    return beta * torch.sum(torch.pow(weight.mm(weight.t()) - torch.eye(weight.shape[0], dtype = torch.float, device = weight.device), 2))
'''

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
Two helper functions for normalizing fc layers and conv layers
'''

'''
def normalize_fc(fc, C):
    norm = torch.norm(fc.weight)
    if norm > C:
        fc.weight /= (norm / C)

def normalize_conv(conv, C):
    assert(0)
    weight = conv.weight.to(torch.device('cpu')).detach().numpy()
    bias = conv.bias.to(torch.device('cpu')).detach().numpy()

    norm = np.linalg.norm(weight.reshape((weight.shape[0], -1)), ord = p)
    norm = np.float(norm)
    
    conv.weight /= norm
    conv.bias /= norm
'''

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
    
    ''' 
    def normalize(self, C = 1):
        normalize_fc(self.fc, C = C)
    '''

class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = None
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 10)
    
    '''
    def __forward__(self, x):
        x = x.view((-1, 784))
        x = F.relu(self.fc1(x))
        return x
    '''

    def forward(self, x):
        x = x.view((-1, 784))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    '''
    def normalize(self):
        normalize_fc(self.fc1)
        normalize_fc(self.fc2)

    def getFinalLayer(self):
        return self.fc2.weight.to(torch.device('cpu')).detach().numpy(), \
               self.fc2.bias.to(torch.device('cpu')).detach().numpy()
    '''

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.fc1 = nn.Linear(12 * 12 * 32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
    
    '''
    def __forward__(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = x.view(-1, 12 * 12 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return x
    '''

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
    def normalize(self):
        normalize_conv(self.conv1)
        normalize_conv(self.conv2)
        normalize_conv(self.conv3)
        normalize_conv(self.conv4)

        normalize_fc(self.fc1)
        normalize_fc(self.fc2)
        normalize_fc(self.fc3)

    def getFinalLayer(self):
        return self.fc3.weight.to(torch.device('cpu')).detach().numpy(), \
               self.fc3.bias.to(torch.device('cpu')).detach().numpy()
    '''  

class CIFARLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = None
        self.fc = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.fc(x)
        return x
    ''' 
    def normalize(self):
        normalize_fc(self.fc)
    '''

'''
CIFAR Models
'''
class CIFARMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 10)
    
    '''
    def __forward__(self, x):
        x = x.view((-1, 3 * 32 * 32))
        #x = F.softplus(self.fc1(x))
        x = F.relu(self.fc1(x))
        return x
    '''

    def forward(self, x):
        x = x.view((-1, 3 * 32 * 32))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    '''
    def normalize(self):
        normalize_fc(self.fc1)
        normalize_fc(self.fc2)

    def getFinalLayer(self):
        return self.fc2.weight.to(torch.device('cpu')).detach().numpy(), \
               self.fc2.bias.to(torch.device('cpu')).detach().numpy()
    '''

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
    
    '''
    def __forward__(self, x):
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
        return x
    '''

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
    
    '''
    def normalize(self):
        normalize_conv(self.conv1)
        normalize_conv(self.conv2)
        normalize_conv(self.conv3)
        normalize_conv(self.conv4)
        
        normalize_fc(self.fc1)
        normalize_fc(self.fc2)
        normalize_fc(self.fc3)

    def getFinalLayer(self):
        return self.fc3.weight.to(torch.device('cpu')).detach().numpy(), \
               self.fc3.bias.to(torch.device('cpu')).detach().numpy()
    '''

