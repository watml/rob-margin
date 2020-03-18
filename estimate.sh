#!/bin/bash

python estimateLipschitzBound.py MNISTLR ./Model/MNISTLR/MNISTLR_00010.tar ./Output/MNISTLR/MNISTLR_00010.pkl MNIST -train 0 -n 500 -Nb 1 -Ns 1 -R 5 -p 2 -ckpt 1

