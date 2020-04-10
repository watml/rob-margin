'''
A script to test accuracy of neural networks.
'''

import torch
from utils import *
import argparse
import os
import copy
import scipy.io as sio

def filt_state_dict(state_dict):
    ret = copy.deepcopy(state_dict)
    for key in state_dict:
        if 'inverse' in key:
            ret.pop(key)
        elif 'net' in key:
            ret[key[4:]] = ret.pop(key)
    return ret

def filt_model(model, device):
    if model.__class__.__name__ == 'MNISTLeNet':
        imdb = sio.loadmat('./deepdefense.pytorch/data/mnist-data-0208ce21.mat')
        mean = imdb['images'][0][0][1].transpose()
        model.mean = torch.tensor(mean, dtype = torch.float, device = device)

    if model.__class__.__name__ == 'CIFARConvNet':
        return

        cifar = makeDataset('CIFAR10')

        imdb = sio.loadmat('./deepdefense.pytorch/data/cifar10-data-ce5d97dd.mat')
        images = imdb['images'][0][0][0].transpose()
        sets = imdb['images'][0][0][2].flatten()
        trainval_idx = np.where(sets == 1)[0]

        images = images[trainval_idx]
        print(images.shape)
        print('min value in images %f' % (np.min(images)))
        print('max value in images %f' % (np.max(images)))
        mean = np.mean(images, axis = 0)
        model.mean = torch.tensor(mean, dtype = torch.float, device = device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('modelname')
    parser.add_argument('path')
    parser.add_argument('dataset')
    parser.add_argument('-ckpt', type = int, default = 0)

    parser.add_argument('-epsilon', type = float, default = 0)
    parser.add_argument('-nb_iter', type = int, default = 40)
    parser.add_argument('-step_size', type = float, default = 0.01)

    parser.add_argument('-preprocess', type = int, default = 0)

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(args)
    print('Test %s on %s' % (args.modelname, device))

    _, testloader = makeLoader(args.dataset, batch_size = 1024)

    model = modelname2model(args.modelname)

    # print(model)

    if args.path.endswith('.mat'):
        load_LeNet_small(model, args.path)
    if bool(args.ckpt) == False:
        state_dict = torch.load(args.path)
        state_dict = filt_state_dict(state_dict)
        model.load_state_dict(state_dict)
    else:
        checkpoint = torch.load(args.path, map_location = 'cpu')
        state_dict = checkpoint['model_state_dict']
        state_dict = filt_state_dict(state_dict)
        model.load_state_dict(state_dict)

    
    if bool(args.preprocess):
        filt_model(model, device)
    model.to(device).eval()

    # print(model)
    
    if args.epsilon < 1e-10:
        print(acc(model, device, testloader))
    else:
        print(robust_acc(model, device, testloader, epsilon = args.epsilon, nb_iter = args.nb_iter, step_size = args.step_size))

if __name__ == '__main__':
    main()

