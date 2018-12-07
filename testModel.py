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

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Test %s on %s' % (args.modelname, device))

    _, testloader = makeLoader(args.dataset, batch_size = 1024)

    model = modelname2model(args.modelname)
    
    if os.path.isfile(args.path) == True:
        model.load_state_dict(torch.load(args.path))
    else:
        assert(0)

    model.eval()
    model.to(device)

    print(acc(model, device, testloader))

if __name__ == '__main__':
    main()
