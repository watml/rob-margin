#!/bin/bash

python trainModel.py CIFARLR ./Model/CIFARLR.pt CIFAR -batch 32 -epochs 10 -lr 0.01 -verbose 2

#python trainModel.py MNISTLR MNIST -augmentation 0 -batch 32 -epochs 10 -lr 0.01 -decay 1.0

#python trainModel.py CIFARLR CIFAR -augmentation 0 -batch 64 -epochs 20 -lr 0.0001

#python estimateDistance.py MNISTLR MNIST -n 1 -Nb 1 -Ns 1024 -R 5 -random 0 -p 2

#python estimateDistance.py MNISTMLP MNIST -n 500 -Nb 1 -Ns 1024 -R 5 -random 0 -p 2

#python estimateDistance.py MNISTCNN MNIST -n 1 -Nb 10 -Ns 1024 -R 5 -random 0 -p 2

#python estimateLinearDistance.py MNISTLR MNIST -n 500 -p 2

#python evaluateModel.py MNISTMLP MNIST

#python trainModel.py MNISTMLP2 MNIST -batch 32 -epochs 50 -lr 0.005
#python estimateDistance.py MNISTMLP2 MNIST -n 1 -Nb 10 -Ns 1024 -R 5 -random 0 -p 2
