'''
A script to test accuracy of neural networks.
'''

import torch
from utils import *
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('modelname')
    parser.add_argument('path')
    parser.add_argument('dataset')
    parser.add_argument('-ckpt', type = int, default = 0)

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Test %s on %s' % (args.modelname, device))

    _, testloader = makeLoader(args.dataset, batch_size = 1024)

    model = modelname2model(args.modelname)
    
    orthogonal_constraint(model, beta = 1)

    if bool(args.ckpt) == False:
        model.load_state_dict(torch.load(args.path))
    else:
        checkpoint = torch.load(args.path)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device).eval()

    print(acc(model, device, testloader))

if __name__ == '__main__':
    main()
