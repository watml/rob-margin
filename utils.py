'''
Some helper functions to train neural networks.
'''
import numpy as np

import torch

import torchvision
import torchvision.transforms as transforms

from model import *

import os

def acc(model, device, loader):
    '''
    Return the accuracy of a given model, returning the model in eval mode on the device.
    '''

    correct = 0
    total = 0

    model.to(device).eval()

    with torch.no_grad():
        
        for img, label in loader:
            
            img, label = img.to(device), label.to(device)
            
            output = model(img)
            
            total += label.shape[0]
            correct += (torch.argmax(output, dim = 1) == label).sum().item()
    
    return correct / total

def train(model, device, trainloader, testloader, loss_fn, optimizer, epochs = 1, verbose = 0, ckpt_folder = None, regularizer = False, mu = 1, tau = 1, beta = 1):
    '''
    Train a model, returning the model in train mode on the device.

    Value of verbose:
    0 -- Only print training loss
    1 -- Print training loss and training acc
    2 -- Print training loss, training error and test acc

    If ckpt_path != None, then assume verbosr = 2
    '''

    print('Train %s on %s' % (model.__class__.__name__, device))
    
    model.to(device).train()

    for i in range(epochs):
        
        total_loss = 0
        
        # first calculate the gradient of H, then apply sgd to G
        if 'CIFAR' not in model.__class__.__name__ and regularizer == True:

            total_loss_H = 0
            
            for img, label in trainloader:
                img, label = img.to(device), label.to(device)
                output = model(img)
                
                n_samples = output.shape[0]
                index = torch.arange(n_samples, device = device)
                output = output[index, label].reshape((n_samples, 1)) - output
                output[index, label] = torch.tensor(1e10, device = device)
                alpha, _ = torch.min(output, dim = 1)
                # regularizer is H_tau(\alpha) - H_0(\alpha)
                loss_H = mu * torch.mean(torch.max(torch.tensor(0, dtype = torch.float, device = device), 0 - alpha))
                loss_H.backward()
                
                total_loss_H += loss_H.item()
            
            total_loss_H /= len(trainloader)

            # resotre the gradient
            params_list = list(model.parameters())
            dH_list = []
            for tensor in params_list:
                dH_list.append(torch.tensor(tensor.grad / len(trainloader), device = device))

        for img, label in trainloader:
            # label is a tensor, one number for each image
            img, label = img.to(device), label.to(device)
            
            model.zero_grad()

            output = model(img)
            loss = loss_fn(output, label)
            
            if regularizer == True:
                n_samples = output.shape[0]
                index = torch.arange(n_samples, device = device)
                output = output[index, label].reshape((n_samples, 1)) - output
                output[index, label] = torch.tensor(1e10, device = device)
                alpha, _ = torch.min(output, dim = 1)
                # regularizer is H_tau(\alpha) - H_0(\alpha)
                if model.__class__.__name__ != 'CIFARCNN':
                    loss += mu * torch.mean(torch.max(torch.tensor(0, dtype = torch.float, device = device), tau - alpha))
                else:
                    loss += mu * torch.mean(torch.clamp(- alpha, min = 0, max = tau))

            if beta > 0:
                loss += orthogonal_constraint(model, device = device, beta = beta)

            loss.backward()

            if 'CIFAR' not in model.__class__.__name__ and regularizer == True:
                # subtract gradient with dH
                with torch.no_grad():
                    params_list = list(model.parameters())
                    for j in range(len(params_list)):
                        params_list[j].grad -= dH_list[j]
            
            optimizer.step()
            
            total_loss += loss.item()

        total_loss /= len(trainloader)
        
        if 'CIFAR' not in model.__class__.__name__ and regularizer == True:
            total_loss -= total_loss_H

        # Start to evaluate model
        model.eval()
        
        if ckpt_folder != None:
            verbose = 2
        
        if verbose == 0:
            print('Epoch : %d, Loss : %.10f' % (i + 1, total_loss))
        elif verbose == 1:
            train_acc = acc(model, device, trainloader)
            print('Epoch : %d, Loss : %.10f, Training Acc : %f' % (i + 1, total_loss, train_acc))
        elif verbose == 2:
            train_acc = acc(model, device, trainloader)
            test_acc = acc(model, device, testloader)
            print('Epoch : %d, Loss : %.10f, Training Acc : %f, Test Acc : %f' % (i + 1, total_loss, train_acc, test_acc))
        else:
            assert(0)
        
        if ckpt_folder != None:
            checkpoint = {'epochs' : i + 1, \
                          'loss' : total_loss, \
                          'train_acc' : train_acc, \
                          'test_acc' : test_acc, \
                          'model_state_dict' : model.state_dict(), \
                          'optmizer_state_dict' : optimizer.state_dict(), \
                          }
            torch.save(checkpoint, ckpt_folder + '/' + model.__class__.__name__ + '_' + ('' if regularizer == False and beta < 1e-15 else ('reg_' if regularizer == 1 else 'constraint_')) + str(i + 1).zfill(5) + '.tar')
        
        # Set model back to train mode for the next epoch
        model.train()

    return model

def makeDataset(dataset, augmentation = False):
    '''
    Take a string as input and output the dataset.
    '''
    if augmentation == True:
        assert(0)

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
    print(args)

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

def __distance__(w, b, x, q):
    '''
    Calculate the distance for a single data point.
    w is d * 1
    b is 1 * 1
    x is d * 1
    '''
    return np.abs(np.dot(w.T, x) + b) / np.linalg.norm(w.squeeze(), ord = q)

def distance(W, b, x, q):
    '''
    Calculate the distance for a multiclass linear classifier.
    W is K * d, where K >= 2
    b is K * 1
    x is d * 1

    ret is scalar
    '''
    K, d = W.shape
    assert(x.shape == (d, 1))
    assert(b.shape == (K, 1))

    c = np.argmax(np.dot(W, x) + b)
    
    ret = np.inf
    for i in range(0, K):
        if i == c:
            continue
        temp = __distance__((W[i, :] - W[c, :]).reshape((d, 1)), b[i, 0] - b[c, 0], x, q = q)
        ret = min(ret, temp)

    return ret

