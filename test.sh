#!/bin/bash

# test model accuracy
# python testModel.py MNISTMLP ./Model/MNISTMLP/MNISTMLP_00200.tar MNIST -ckpt 1
# python testModel.py MNISTMLP ./Model/MNISTMLP_constraint_00200.tar MNIST -ckpt 1
# python testModel.py MNISTMLP ./Model/MNISTMLP_reg_00200.tar MNIST -ckpt 1

# python testModel.py MNISTCNN ./Model/MNISTCNN_00200.tar MNIST -ckpt 1
# python testModel.py MNISTCNN ./Model/MNISTCNN_constraint_00200.tar MNIST -ckpt 1
# python testModel.py MNISTCNN ./Model/MNISTCNN_reg_00200.tar MNIST -ckpt 1

# python testModel.py CIFARMLP ./Model/CIFARMLP/CIFARMLP_00200.tar CIFAR -ckpt 1
# python testModel.py CIFARMLP ./Model/CIFARMLP_constraint_00200.tar CIFAR -ckpt 1
# python testModel.py CIFARMLP ./Model/CIFARMLP_reg_00200.tar CIFAR -ckpt 1

# python testModel.py CIFARCNN ./Model/CIFARCNN_00200.tar CIFAR -ckpt 1
# python testModel.py CIFARCNN ./Model/CIFARCNN_constraint_00200.tar CIFAR -ckpt 1
# python testModel.py CIFARCNN ./Model/CIFARCNN_reg_00200.tar CIFAR -ckpt 1

# test model under PGD attack
# python testModel.py MNISTMLP ./Model/MNISTMLP_00200.tar MNIST -ckpt 1 -epsilon 1.0 -nb_iter 1000 -step_size 0.01
# python testModel.py MNISTMLP ./Model/MNISTMLP_constraint_00200.tar MNIST -ckpt 1 -epsilon 1.0 -nb_iter 1000 -step_size 0.01
# python testModel.py MNISTMLP ./Model/MNISTMLP_reg_00200.tar MNIST -ckpt 1 -epsilon 1.0 -nb_iter 1000 -step_size 0.01

# python testModel.py MNISTCNN ./Model/MNISTCNN_00200.tar MNIST -ckpt 1 -epsilon 2.0 -nb_iter 1000 -step_size 0.01
# python testModel.py MNISTCNN ./Model/MNISTCNN_constraint_00200.tar MNIST -ckpt 1 -epsilon 1.0 -nb_iter 1000 -step_size 0.01
# python testModel.py MNISTCNN ./Model/MNISTCNN_reg_00200.tar MNIST -ckpt 1 -epsilon 2.0 -nb_iter 1000 -step_size 0.01

# python testModel.py CIFARCNN ./Model/CIFARCNN_00200.tar CIFAR -ckpt 1 -epsilon 0.4 -nb_iter 1000 -step_size 0.01
# python testModel.py CIFARCNN ./Model/CIFARCNN_constraint_00200.tar CIFAR -ckpt 1 -epsilon 0.4 -nb_iter 1000 -step_size 0.01
# python testModel.py CIFARCNN ./Model/CIFARCNN_reg_00200.tar CIFAR -ckpt 1 -epsilon 0.4 -nb_iter 1000 -step_size 0.01
