'''
A script to train neural networks.
'''

import torch
import torch.optim as optim
import torch.nn.functional as F
import os

from utils import *

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('modelname')
    parser.add_argument('path', help = 'File name: ./Model/MNISTLR.pt, Folder name: ./Model/MNISTLR/')
    
    parser.add_argument('dataset')
    parser.add_argument('-augmentation', type = int, default = 0, help = 'Deprecated, one should never use it.')
    
    parser.add_argument('-batch', type = int, default = 32)
    parser.add_argument('-epochs', type = int, default = 1)
    parser.add_argument('-lr', type = float, default = 0.01)
    parser.add_argument('-decay', type = float, default = 0.0, help = 'Deprecated, one should never use it.')
    parser.add_argument('-momentum', type = float, default = 0.0)
    parser.add_argument('-nesterov', type = int, default = 0)
    parser.add_argument('-verbose', type = int, default = 2)
    
    parser.add_argument('-ckpt', type = int, default = 0, help = 'If ckpt == false, then path is a file, otherwise path is a folder, storing all checkpoints during training.')
    
    parser.add_argument('-reg', type = int, default = 0)
    parser.add_argument('-mu', type = float, default = 1.0)
    parser.add_argument('-tau', type = float, default = 1.0)
    parser.add_argument('-beta', type = float, default = 1.0)

    parser.add_argument('-adv', type = int, default = 0)
    parser.add_argument('-epsilon', type = float, default = 0.3)
    parser.add_argument('-nb_iter', type = int, default = 40)
    parser.add_argument('-step_size', type = float, default = 0.01)

    args = parser.parse_args()
    
    printArguments(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trainloader, testloader = makeLoader(args.dataset, batch_size = args.batch, augmentation = bool(args.augmentation))
    
    model = modelname2model(args.modelname)
    optimizer = optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.decay, momentum = args.momentum, nesterov = bool(args.nesterov))

    if bool(args.ckpt) == 0:
        # Save a single model
        model.to(device)
        model = train(model, device, trainloader, testloader, loss_fn = F.cross_entropy, optimizer = optimizer, epochs = args.epochs, verbose = args.verbose, ckpt_folder = None, \
                      adv = bool(args.adv), epsilon = args.epsilon, nb_iter = args.nb_iter, step_size = args.step_size, \
                      regularizer = bool(args.reg), mu = args.mu, tau = args.tau, beta = args.beta)

        torch.save(model.state_dict(), args.path)
    else:
        # Save all models during training
        if os.path.isdir(args.path) == False:
            os.mkdir(args.path)
            
        model.to(device)
        torch.save({'epoch' : 0, \
                    'loss' : 0, \
                    'train_acc' : acc(model, device, trainloader), \
                    'test_acc' : acc(model, device, testloader), \
                    'model_state_dict' : model.state_dict(), \
                    'optimizer_state_dict' : optimizer.state_dict(), \
                    }, args.path + '/' + args.modelname + '_' + ('' if bool(args.reg) == False and args.beta < 1e-15 else ('reg_' if bool(args.reg) ==  1  else 'constraint_')) + str(0).zfill(5) + '.tar')

        model.to(device)
        model = train(model, device, trainloader, testloader, loss_fn = F.cross_entropy, optimizer = optimizer, epochs = args.epochs, verbose = args.verbose, ckpt_folder = args.path, \
                      adv = args.adv, epsilon = args.epsilon, nb_iter = args.nb_iter, step_size = args.step_size, \
                      regularizer = bool(args.reg), mu = args.mu, tau = args.tau, beta = args.beta)

if __name__ == '__main__':
    main()

