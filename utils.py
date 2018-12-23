'''
Some helper functions to train neural networks.
'''
import numpy as np

import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from MNISTModel import *
from CIFARModel import *

import os

def acc(model, device, loader):
    '''
    Return the accuracy of a given model, assuming that model is in the eval mode, returning the model in eval mode.
    '''

    correct = 0
    total = 0

    with torch.no_grad():
        
        for img, label in loader:
            
            img, label = img.to(device), label.to(device)
            
            output = model(img)
            
            total += label.shape[0]
            correct += (torch.argmax(output, dim = 1) == label).sum().item()
    
    return correct / total

def train(model, device, trainloader, testloader, loss_fn, optimizer, epochs = 1, verbose = 0):
    '''
    Train a model, assuming that the model is in train mode, returning the model in train mode.

    Value of verbose:
    0 -- Only print training loss
    1 -- Print training loss and training acc
    2 -- Print training loss, training error and test acc

    One should move model to the device before calling this function.
    '''

    # print('Train %s on %s' % (model.__class__.__name__, device))

    for i in range(epochs):
        
        total_loss = 0
        
        for img, label in trainloader:
            # label is a tensor, one number for each image        
            img, label = img.to(device), label.to(device)
            
            model.zero_grad()

            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(trainloader)
        
        model.eval()
        
        if verbose == 0:
            print('Epoch : %d, Loss : %f' % (i + 1, total_loss))
        elif verbose == 1:
            print('Epoch : %d, Loss : %f, Training Acc : %f' % (i + 1, total_loss, acc(model, device, trainloader)))
        elif verbose == 2:
            print('Epoch : %d, Loss : %f, Training Acc : %f, Test Acc : %f' % (i + 1, total_loss, acc(model, device, trainloader), acc(model, device, testloader)))
        else:
            assert(0)

        model.train()

    return model

def trainSavedModel(path, model, device, trainloader, testloader, loss_fn, optimizer, epochs = 1, verbose = 0):
    '''
    Train a saved model.
    '''

    if os.path.isfile(path) == True:
        model.load_state_dict(torch.load(path))
    
    model.to(device)
    
    model.train() 
    train(model, device, trainloader, testloader, loss_fn, optimizer, epochs, verbose)
    model.eval()

    print('Model Performance : Training Acc : %f, Test Acc : %f' % (acc(model, device, trainloader), acc(model, device, testloader)))

    # Always save model in cpu version
    model.to(torch.device('cpu'))
    torch.save(model.state_dict(), path)

def makeDataset(dataset, augmentation = False):
    '''
    Take a string as input and output the dataset.
    '''

    if augmentation == False:
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
    else:
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding = 4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
    
    if dataset == 'MNIST':
        trainingset = torchvision.datasets.MNIST(root = './Data', train = True, download = True, transform = transform_train)
        testset = torchvision.datasets.MNIST(root = './Data', train = False, download = True, transform = transform_test)
    elif dataset == 'CIFAR':
        trainingset = torchvision.datasets.CIFAR10(root = './Data', train = True, download = True, transform = transform_train)
        testset = torchvision.datasets.CIFAR10(root = './Data', train = False, download = True, transform = transform_test)
    else:
        assert(0)

    return trainingset, testset


def makeLoader(dataset, batch_size = 64, augmentation = False):
    '''
    Take a dataset as input and output the data loader.
    Dataset can be a string, e.g. 'MNIST' and 'CIFAR', or a tuple of trainingset and testset.
    '''

    if isinstance(dataset, str):
        trainingset, testset = makeDataset(dataset, augmentation)
    else:
        trainingset, testset = dataset
    
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size = batch_size, shuffle = True, num_workers = 8)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 8)

    return trainloader, testloader


def modelname2model(modelname):
    '''
    Given a string, return the corresponding model object.
    '''

    modelname_dict = { \
        'MNISTLR': MNISTLR(), \
        'MNISTMLP': MNISTMLP(), \
        'MNISTCNN': MNISTCNN(), \
        'CIFARLR': CIFARLR(), \
        'CIFARMLP': CIFARMLP(), \
        'CIFARCNN': CIFARCNN() \
    }

    model = modelname_dict.get(modelname)

    if model == None:
        assert(0)
    else:
        return model

def printArguments(args):
    '''
    Print every entry of args
    '''

    args_dict = vars(args)

    for arg in args_dict:
        print('%s = %s' % (arg, args_dict.get(arg)))

def dual(p):
    '''
    Return the number of dual norm of p.
    '''

    if p == np.inf:
        return 1
    elif p == 1:
        return np.inf
    else:
        return p / (p - 1)

