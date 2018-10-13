'''
A defensive distillation model on MNIST.
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

from MNISTModel import MNISTCNN
from MNISTModel import MNISTMLP

import os.path

from utils import train, acc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 64

transform = transforms.Compose([transforms.ToTensor()])

trainingset = torchvision.datasets.MNIST(root = './Data', train = True, download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainingset, batch_size = batch_size, shuffle = True, num_workers = 8)

testset = torchvision.datasets.MNIST(root = './Data', train = False, download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 8)

teacher_net = MNISTCNN()
teacher_net.to(device)

if os.path.isfile('./Model/teacherNetMNIST.pt') == False:	
    optimizer = optim.SGD(teacher_net.parameters(), lr = 0.01, weight_decay = 1e-6, momentum = 0.9, nesterov = True)
    train(teacher_net, device, trainloader, testloader, loss_fn = F.cross_entropy, optimizer = optimizer, epochs = 1, temperature = 100)
    torch.save(teacher_net.state_dict(), './Model/teacherNetMNIST.pt')
else: 
    teacher_net.load_state_dict(torch.load('./Model/teacherNetMNIST.pt'))
    teacher_net.eval()

print('Teacher Net Performance : Training Acc : %f, Test Acc : %f' % (acc(teacher_net, device, trainloader), acc(teacher_net, device, testloader)))

class SoftLabelMNIST(Dataset):
    def __init__(self, trainingset, oracle, temperature):
        
        softmax = torch.nn.Softmax(dim = 1) 
        
        with torch.no_grad():
            self.X = [trainingset[i][0] for i in range(60000)]
            self.y = []
            for i in range(60000):
                self.y.append(softmax(oracle(self.X[i].reshape((1, 1, 28, 28)).to(device)) / temperature).squeeze())

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        return len(self.X)

def SoftLabelCrossEntropy(output, label):
    
    LogSoftmax = nn.LogSoftmax(dim = 1)
    output = - LogSoftmax(output)
    
    return torch.sum(output * label) / output.shape[0]

student_net = MNISTCNN()
student_net.to(device)

if os.path.isfile('./Model/studentNetMNIST.pt') == False:
    
    print('Constructing data set')
    new_trainingset = SoftLabelMNIST(trainingset, teacher_net, temperature = 100)
    new_trainloader = torch.utils.data.DataLoader(new_trainingset, batch_size = batch_size, shuffle = True, num_workers = 0)
    print('Finish construction')

    optimizer = optim.SGD(student_net.parameters(), lr = 0.01, weight_decay = 1e-6, momentum = 0.9, nesterov = True)
    train(student_net, device, new_trainloader, testloader, SoftLabelCrossEntropy, optimizer = optimizer, epochs = 1, temperature = 100)
            
    torch.save(student_net.state_dict(), './Model/studentNetMNIST.pt')

else:
    student_net.load_state_dict(torch.load('./Model/studentNetMNIST.pt'))
    student_net.eval()

print('Student Net Performance : Training Acc : %f, Test Acc : %f' % (acc(student_net, device, trainloader), acc(student_net, device, testloader)))

cnnMNIST = MNISTCNN().to(device)

if os.path.isfile('./Model/cnnMNIST.pt') == False:
    
    optimizer = optim.SGD(cnnMNIST.parameters(), lr = 0.01, weight_decay = 1e-6, momentum = 0.9, nesterov = True)
    train(cnnMNIST, device, trainloader, testloader, F.cross_entropy, optimizer = optimizer, epochs = 10, temperature = 1)
            
    torch.save(cnnMNIST.state_dict(), './Model/cnnMNIST.pt')

else:
    cnnMNIST.load_state_dict(torch.load('./Model/cnnMNIST.pt'))
    cnnMNIST.eval()

print('cnnMNIST Performance : Training Acc : %f, Test Acc : %f' % (acc(cnnMNIST, device, trainloader), acc(cnnMNIST, device, testloader)))


'''
mlp = MNISTMLP()
mlp.to(device)

if os.path.isfile('mlp.pt') == False:
    loss_fn = nn.CrossEntropyLoss()
    train(mlp, trainloader, testloader, loss_fn = loss_fn, lrate = 0.01, epochs = 1, temperature = 1)
    torch.save(mlp.state_dict(), 'mlp.pt')
    
else:
    mlp.load_state_dict(torch.load('mlp.pt'))
    mlp.eval()
    
print('MLP Net Performance : Training Acc : %f, Test Acc : %f' % (acc(mlp, trainloader), acc(mlp, testloader)))
'''


