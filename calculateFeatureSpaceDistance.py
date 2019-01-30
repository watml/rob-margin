'''
A script to calculate distance in feature space for neural networks.
'''

import torch
import numpy as np

from utils import *

import argparse
import pickle

def calculateFeatureSpaceDistance(model, device, dataset, q):
    model.to(device).eval()
    
    print('Feature space distance for %s on %s' % (model.__class__.__name__, device))

    dist = []
    target = []
    prediction = []
    
    params = list(model.parameters())
    #print(torch.norm(params[0]))
    with torch.no_grad():
        model.normalize()
    #print(torch.norm(params[0]))
    
    weight, bias = model.getFinalLayer()
    assert(len(weight.shape) == 2 and weight.shape[0] == 10)
    assert(bias.shape == (10, ) )
        
    for i in range(len(dataset)):
        img, label = dataset[i]
        img, label = img.to(device), torch.tensor(label, device = device, dtype = torch.long)

        # Reshape the imput to a tensor with batch_size = 1
        x0 = img.view((1, *tuple(img.shape)))
        with torch.no_grad():
            output = model(x0)
            c = torch.argmax(output, dim = 1).item()
        target.append(label.item())
        prediction.append(c)
        
        with torch.no_grad():
            feature = model.__forward__(x0)
        
        dist_x0 = distance(weight, \
                           bias.reshape((-1, 1)), \
                           feature.detach().to(torch.device('cpu')).numpy().reshape((-1, 1)), \
                           q = q)
        dist.append(dist_x0)

    return dist, target, prediction

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('modelname')
    parser.add_argument('model_path')
    parser.add_argument('store_path')
    parser.add_argument('dataset')

    parser.add_argument('-train', type = int, default = 0, help = 'Evaluate on training set or test set.')
    parser.add_argument('-n', type = int, default = 500, help = 'Number of samples that need to estimate.')
    parser.add_argument('-p', type = int, default = 2)
    parser.add_argument('-ckpt', type = int, default = 0, help = 'Indicate whether model path is a check point or not.')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the pretrained model.
    model = modelname2model(args.modelname)
    
    if bool(args.ckpt) == False:
        model.load_state_dict(torch.load(args.model_path))
    else:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    p = args.p if args.p < 1e10 else np.inf
    q = dual(p)

    # Prepare the subset that need to estimate. Fix the random seed to make sure the subset is the same every time.
    trainingset, testset = makeDataset(args.dataset)
    
    np.random.seed(0)
    
    if bool(args.train) == True:
        index = np.random.choice(len(trainingset), size = args.n, replace = False)
        subset = [trainingset[i] for i in index]
    else:
        index = np.random.choice(len(testset), size = args.n, replace = False)
        subset = [testset[i] for i in index]

    # Reset the random seed.
    np.random.seed()

    printArguments(args)
    
    dist, target, prediction = calculateFeatureSpaceDistance(model, device, subset, q)
    
    estimation = {'index' : index, \
                  'target' : target, \
                  'prediction' : prediction, \
                  'dist' : dist, \
                  'config' : args, \
                  }

    print(np.mean(dist))
    
    torch.save(estimation, args.store_path)

if __name__ == '__main__':
    main()

