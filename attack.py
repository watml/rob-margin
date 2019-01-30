'''
Implementation of several attack methods.
'''

import numpy as np
import torch
import argparse
import pickle

from utils import *
from utils import __distance__

def attack4linear(W, b, x, q):
    '''
    Optimal attack for linear classifiers.
    W is K * d, where K >= 2
    b is K * 1
    x is d * 1

    return the optimal adversarial example and the distance to clean example.
    '''
    K, d = W.shape
    assert(x.shape == (d, 1))
    assert(b.shape == (K, 1))

    c = np.argmax(np.dot(W, x) + b)
    
    ret = np.inf
    index = 0
    for i in range(0, K):
        if i == c:
            continue
        temp = __distance__((W[i, :] - W[c, :]).reshape((d, 1)), b[i, 0] - b[c, 0], x, q = q)
        if temp < ret: 
            ret = temp
            index = i
    
    w = (W[c, :] - W[index, :]).reshape((d, 1))
    b = b[c, 0] - b[index, :]
    
    return x - ((np.dot(w.T, x) + b) / np.dot(w.T, w) + 1e-6) * w, ret

'''
def attackLinearModel(model, x, q):
    W, b = list(model.parameters())
    W = W.to(torch.device('cpu')).detach().numpy()
    b = b.to(torch.device('cpu')).detach().numpy().reshape((10, 1))

    return attack4linear(W, b, x, q = q)
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('modelname')
    parser.add_argument('folder')
    parser.add_argument('dataset')
    parser.add_argument('-ckpt', type = int, default = 0)

    args = parser.parse_args()

    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    print('Attack %s on %s' % (args.modelname, device))

    trainingset, testset = makeDataset(args.dataset)
    
    model_list = []
    index_list = [1, 10, 20, 50, 80, 100, 200, 500, 800, 1000]
    for i in range(len(index_list)):
        model = modelname2model(args.modelname)

        checkpoint = torch.load(args.folder + args.modelname + '_' + str(index_list[i]).zfill(5) + '.tar')
        model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device).eval()
        model_list.append(model)
    
    clean_img_list = []
    adv_img_list = []
    for i in range(10):
        temp = [img_label for img_label in testset if img_label[1].item() == i]
        img, label = temp[0]
        
        x0 = img.to(torch.device('cpu')).detach().numpy().reshape((img.numel(), 1))
        label = label.item()
        
        adv_img_epoch_list = []

        for model in model_list:
            W, b = list(model.parameters())
            W = W.to(torch.device('cpu')).detach().numpy()
            b = b.to(torch.device('cpu')).detach().numpy().reshape((10, 1))
            
            x_adv, _ = attack4linear(W, b, x0, q = 2)
            adv_img_epoch_list.append(x_adv)
            
            assert(np.argmax(np.dot(W, x0) + b) != np.argmax(np.dot(W, x_adv) + b))
            
            print(_)
            #print(np.argmax(np.dot(W, x0) + b))
            #print(np.argmax(np.dot(W, x_adv) + b))
            #print(np.dot(W, x_adv) + b)
            
        print('****************')
        clean_img_list.append(x0)
        adv_img_list.append(adv_img_epoch_list)
        
    with open('./Output/MNISTLR_Adv_Visualization.pkl', 'wb') as f:
        pickle.dump((clean_img_list, adv_img_list), f)

if __name__ == '__main__':
    main()

