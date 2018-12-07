#!/bin/bash


#python estimateLinearDistance.py MNISTLR MNIST -n 500 -p 2

python estimateLinearDistance.py CIFARLR CIFAR -n 500 -p 2

#python evaluateModel.py MNISTMLP MNIST

#python trainModel.py MNISTMLP2 MNIST -batch 32 -epochs 50 -lr 0.005
#python estimateDistance.py MNISTMLP2 MNIST -n 1 -Nb 10 -Ns 1024 -R 5 -random 0 -p 2
