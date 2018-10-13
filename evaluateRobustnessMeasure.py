'''
Compute different measure based sampling results.
'''

import numpy as np
import matplotlib as mlp
mlp.use('tkagg')
from matplotlib import pyplot as plt

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('modelname')
parser.add_argument('-t', type = float, default = 1.0)

args = parser.parse_args()
print('Measure robustness of %s' % (args.modelname))

# Read required data
index = np.genfromtxt('./Output/' + args.modelname + '/' + args.modelname + '_Index.csv')
target = np.genfromtxt('./Output/' + args.modelname + '/' + args.modelname + '_Target.csv')
prediction = np.genfromtxt('./Output/' + args.modelname + '/' + args.modelname + '_Prediction.csv')
dist = np.genfromtxt('./Output/' + args.modelname + '/' + args.modelname + '_EstimatedDistance.csv')

def acc(prediction, target):
    return np.mean(prediction == target)

def average_distance(dist):
    return np.mean(dist)

def truncated_average_distance(dist, tau):
    return np.mean(np.minimum(dist, tau))

def adversarial_frequency(dist, tau):
    return np.mean(dist <= tau)

def adversarial_severity(dist, tau):
    index = dist <= tau
    return np.mean(dist[index])

def label_average_distance(dist, prediction, target):
    return np.mean(dist * (prediction == target))

def truncated_label_average_distance(dist, tau, prediction, target):
    return np.mean(np.minimum(dist, tau) * (prediction == target))

def label_conditional_average_distance(dist, prediction, target):
    return np.mean(dist[prediction == target])

def truncated_label_conditional_average_distance(dist, tau, prediction, target):
    return np.mean(np.minimum(dist[prediction == target], tau))

tau = args.t

print('Adversarial Frequency: %f' % adversarial_frequency(dist, tau))
print('Adversarial Severity: %f' % adversarial_severity(dist, tau))

print('******************')

print('Average Distance: %f' % average_distance(dist))
print('Label Average Distance: %f' % label_average_distance(dist, prediction, target))
print('Label Conditional Average Distance: %f' % label_conditional_average_distance(dist, prediction, target))

print('******************')

print('Truncated Average Distance: %f' % truncated_average_distance(dist, tau))
print('Truncated Label Average Distance: %f' % truncated_label_average_distance(dist, tau, prediction, target))
print('Truncated Label Conditional Average Distance: %f' % truncated_label_conditional_average_distance(dist, tau, prediction, target))

plt.figure()
plt.hist(dist, bins = 50)
plt.show()
