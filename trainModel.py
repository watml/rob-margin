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

    args = parser.parse_args()
    
    printArguments(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trainloader, testloader = makeLoader(args.dataset, batch_size = args.batch, augmentation = bool(args.augmentation))
    
    model = modelname2model(args.modelname)
    optimizer = optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.decay, momentum = args.momentum, nesterov = bool(args.nesterov))

    if bool(args.ckpt) == 0:
        # Save a single model
        train(model, device, trainloader, testloader, loss_fn = F.cross_entropy, optimizer = optimizer, epochs = args.epochs, verbose = args.verbose)

        model.to(torch.device('cpu'))
        torch.save(model.state_dict(), args.path)
    else:
        # Save all models during training
        if os.path.isdir(args.path) == False:
            os.mkdir(args.path)

        model.to(torch.device('cpu'))
        torch.save({'epoch' : 0, \
                    'loss' : 0, \
                    'train_acc' : acc(model, torch.device('cpu'), trainloader), \
                    'test_acc' : acc(model, torch.device('cpu'), testloader), \
                    'model_state_dict' : model.state_dict(), \
                    'optimizer_state_dict' : optimizer.state_dict(), \
                    }, args.path + '/' + args.modelname + '_' + str(0).zfill(5) + '.tar')

        model.to(device)
        model = train(model, device, trainloader, testloader, loss_fn = F.cross_entropy, optimizer = optimizer, epochs = args.epochs, verbose = args.verbose, ckpt_folder = args.path)

if __name__ == '__main__':
    main()
