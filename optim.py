#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date: 2017-03-15
"""

import theano
import theano.tensor as T
from collections import OrderedDict
from methods import clip, scale
import numpy as np


def sgd(loss, params, learning_rate, clip_at=5.0, scale_norm=0.0):

    updates = OrderedDict()
    grads = T.grad(cost=loss, wrt=params)
    epsilon = 1e-8

    for p, g in zip(params, grads):
        # if clip_at > 0.0:
        #     grad = clip(g, clip_at)
        # else:
        #     grad = g
        #
        # if scale_norm > 0.0:
        #     grad = scale(grad, scale_norm)
        grad_norm = g.norm(L=2)
        grad = (T.minimum(clip_at, grad_norm) / (grad_norm + epsilon)) * g

        updates[p] = p - learning_rate * grad
    return updates, grads


def sgd_momentum(loss, params, learning_rate=1e-1, clip_at=5.0, scale_norm=0.0):
    updates = OrderedDict()
    grads = T.grad(cost=loss, wrt=params)

    momentum=0.9

    for p, g in zip(params, grads):
        c = theano.shared(np.zeros_like(p.get_value(borrow=True)))
        if clip_at > 0.0:
            grad = clip(g, clip_at)
        else:
            grad = g

        if scale_norm > 0.0:
            grad = scale(grad, scale_norm)

        v = momentum * c - learning_rate * grad
        updates[p] = p + v

    return updates, grads
