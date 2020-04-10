#!/bin/bash

# python trainModel.py MNISTLR ./Model/MNISTLR MNIST -batch 128 -epochs 1000 -lr 0.01 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 0 -beta 0.0
# python trainModel.py MNISTLR ./Model/MNISTLR_Constraint MNIST -batch 128 -epochs 1000 -lr 0.01 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 0
# python trainModel.py MNISTLR ./Model/MNISTLR_Reg MNIST -batch 128 -epochs 1000 -lr 0.002 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 1 -mu 1e-1 -tau 5 -beta 1e-3

# python trainModel.py MNISTMLP ./Model/MNISTMLP MNIST -batch 128 -epochs 200 -lr 0.01 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 0 -beta 0.0
# python trainModel.py MNISTMLP ./Model/MNISTMLP_Constraint MNIST -batch 128 -epochs 200 -lr 0.01 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 0 -mu 1e-1 -tau 5 -beta 1e-3
# python trainModel.py MNISTMLP ./Model/MNISTMLP_Reg MNIST -batch 128 -epochs 200 -lr 0.002 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 1 -mu 1e-1 -tau 5 -beta 1e-3

# python trainModel.py MNISTCNN ./Model/MNISTCNN MNIST -batch 128 -epochs 200 -lr 0.01 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 0 -beta 0.0
# python trainModel.py MNISTCNN ./Model/MNISTCNN_Constraint MNIST -batch 128 -epochs 200 -lr 0.01 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 0 -mu 1e-1 -tau 5 -beta 1e-3
# python trainModel.py MNISTCNN ./Model/MNISTCNN_Reg MNIST -batch 128 -epochs 200 -lr 0.002 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 1 -mu 1e-1 -tau 40 -beta 1e-3

# python trainModel.py CIFARLR ./Model/CIFARLR CIFAR -batch 128 -epochs 1000 -lr 0.01 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 0 -beta 0.0
# python trainModel.py CIFARLR ./Model/CIFARLR_Constraint CIFAR -batch 128 -epochs 200 -lr 0.01 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 0 -mu 1e-1 -tau 5 -beta 1e-3
# python trainModel.py CIFARLR ./Model/CIFARLR_Reg CIFAR -batch 128 -epochs 1000 -lr 0.002 -decay 1e-3 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 1 -mu 1e-1 -tau 5 -beta 1e-3

# python trainModel.py CIFARMLP ./Model/CIFARMLP CIFAR -batch 128 -epochs 200 -lr 0.01 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 0 -beta 0.0
# python trainModel.py CIFARMLP ./Model/CIFARMLP_Constraint CIFAR -batch 128 -epochs 200 -lr 0.01 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 0 -mu 1e-1 -tau 10 -beta 1e-3
# python trainModel.py CIFARMLP ./Model/CIFARMLP_Reg CIFAR -batch 128 -epochs 200 -lr 0.002 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 1 -mu 1e-1 -tau 10 -beta 1e-3

# python trainModel.py CIFARCNN ./Model/CIFARCNN CIFAR -batch 128 -epochs 200 -lr 0.005 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 0 -beta 0.0
# python trainModel.py CIFARCNN ./Model/CIFARCNN_constraint CIFAR -batch 128 -epochs 200 -lr 0.01 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 0 -mu 1e-1 -tau 10 -beta 1e-3
# python trainModel.py CIFARCNN ./Model/CIFARCNN_Reg CIFAR -batch 128 -epochs 200 -lr 0.002 -momentum 0.9 -nesterov 1 -verbose 2 -ckpt 1 -reg 1 -mu 1e-1 -tau 10 -beta 1e-3
