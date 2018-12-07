'''
A script to train neural networks.
'''

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import *

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('modelname')
    parser.add_argument('path')
    parser.add_argument('dataset')
    parser.add_argument('-augmentation', type = int, default = 0)
    parser.add_argument('-batch', type = int, default = 64)
    parser.add_argument('-epochs', type = int, default = 1)
    parser.add_argument('-lr', type = float, default = 0.01)
    parser.add_argument('-decay', type = float, default = 0.0)
    parser.add_argument('-momentum', type = float, default = 0.0)
    parser.add_argument('-nesterov', type = int, default = 0)
    parser.add_argument('-verbose', type = int, default = 2)

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Train %s on %s' % (args.modelname, device))

    trainloader, testloader = makeLoader(args.dataset, batch_size = args.batch, augmentation = bool(args.augmentation))
    
    model = modelname2model(args.modelname)

    optimizer = optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.decay, momentum = args.momentum, nesterov = bool(args.nesterov))
    trainSavedModel(args.path, model, device, trainloader, testloader, loss_fn = F.cross_entropy, optimizer = optimizer, epochs = args.epochs, verbose = args.verbose)

if __name__ == '__main__':
    main()