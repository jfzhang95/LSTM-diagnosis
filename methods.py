#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date: 2017-03-05
"""

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def random_weights(shape, name=None):
    return theano.shared(floatX(np.random.uniform(size=shape, low=0.001, high=0.001)), name=name)

def zeros(shape, name=""):
    return theano.shared(floatX(np.zeros(shape)), name=name)

def sigmoid(X):
    return 1 / (1 + T.exp(-X))

def dropout(X, dropout_prob=0.0):
    retain_prob = 1 - dropout_prob
    srng = RandomStreams(seed=1234)
    X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    X /= retain_prob
    return X

def clip(X, epsilon):
    return T.maximum(T.minimum(X, epsilon), -1*epsilon)

def scale(X, max_norm):
    curr_norm = T.sum(T.abs_(X))
    return ifelse(T.lt(curr_norm, max_norm), X, max_norm * (X / curr_norm))


def momentum(loss, params, caches, lr=1e-1, rho=0.1, clip_at=5.0, scale_norm=0.0, reg=0.0):
    updates = OrderedDict()
    grads = T.grad(loss=loss, wrt=params)
    
    caches = list(caches)
    for p, c, g in zip(params, caches, grads):
        if clip_at > 0.0:
            grad = clip(g, clip_at) 
        else:
            grad = g

        if scale_norm > 0.0:
            grad = scale(grad, scale_norm)

        delta = rho * grad + (1-rho) * c
	updates[p] = p - lr * (delta + reg * p)

    return updates, grads

def get_params(layers):
    params = []
    for layer in layers:
        for param in layer.get_params():
            params.append(param)
    return params

def make_caches(params):
    caches = []
    for p in params:
        caches.append(theano.shared(floatX(np.zeros(p.get_value().shape))))
    return caches



