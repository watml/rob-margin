'''
Some helper functions to train neural networks
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path

def acc(model, device, loader):

    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            input, label = data
            input, label = input.to(device), label.to(device)
            output = model(input)
            total += label.size(0)
            correct += (torch.argmax(output, dim = 1) == label).sum().item()
    
    return correct / total

def train(model, device, trainloader, testloader, loss_fn, optimizer, epochs = 1, temperature = 1.0):
 
    for i in range(epochs):
        
        total_loss = 0
        
        for data in trainloader:
	        
            input, label = data
            input, label = input.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(input)
            output = output / temperature
            loss = loss_fn(output, label)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(trainloader)
        
        model.eval()
        print('Epoch : %d, Loss : %f, Test Acc : %f' % (i + 1, total_loss, acc(model, device, testloader)))
        model.train()

def trainSaveModel(filename, model, device, trainloader, testloader, loss_fn, optimizer, epochs = 1, temperature = 1.0):
    
    if os.path.isfile(filename) == True:
        model.load_state_dict(torch.load(filename))
    
    model.to(device)
    model.train()
 
    train(model, device, trainloader, testloader, loss_fn, optimizer, epochs = epochs, temperature = temperature)
    model.eval()
    print('Model Performance : Training Acc : %f, Test Acc : %f' % (acc(model, device, trainloader), acc(model, device, testloader)))

    # Always save model in cpu version
    model.to(torch.device('cpu'))
    torch.save(model.state_dict(), filename)

