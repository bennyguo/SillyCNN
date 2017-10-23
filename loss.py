from __future__ import division
from __future__ import print_function
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return 0.5 * np.mean(np.sum(np.square(input - target), axis=1))

    def backward(self, input, target):
        return (input - target) / len(input)


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name
        self._saved_h = None

    def forward(self, input, target):
        input_exp = np.exp(input)
        h = np.divide(input_exp.T, np.sum(input_exp, 1)).T
        self._saved_h = h
        return np.mean(-np.sum(np.multiply(np.log(h), target), 1))

    def backward(self, input, target):
        h = self._saved_h
        return (h - target) / target.shape[0]
