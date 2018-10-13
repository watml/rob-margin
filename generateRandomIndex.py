'''
Randomly generate index for MNIST and CIFAR
'''

import numpy as np

index = np.random.permutation(10000)

np.savetxt('Index.txt', index, fmt = '%d')


