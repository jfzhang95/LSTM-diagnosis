#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date: 2017-03-05
"""


from LSTM_model import LSTM_Diagnosis
import numpy as np


################################################################################
#   Load data
#   X should be a list of 3d arrays.
#   Y is a 2d array of examples by labels. All entries should be binary.
################################################################################



def run(X, Y, X_test=None, iters=10000, lr=1e-1, alpha=0.0, reg=0.0, dropout_prob=0.5):
    running_total_loss = 0
    print "lr:", lr
    print "alpha:", alpha
    print "reg:", reg
    print "dropout:", dropout_prob
    print "iterations:", iters
    rnn = LSTM_Diagnosis(num_input=6, num_hidden=[128, 128], num_output=500) 

    for iter in xrange(iters):
        total_loss = 0.0
        for i in range(len(X) - 1):
            loss = rnn.train(X[i], np.tile(Y[i], (len(X[i]), 1)), lr, alpha, reg, dropout_prob)
            total_loss += loss
        print "iteration: %s, loss: %s" % (iter, total_loss)

    Y_test = rnn.predict(X_test)
    print "Y_test:", Y_test


shape1 = (50, 25, 13)
X = np.random.randn(*shape1)
shape2 = (50, 9)
Y = np.random.randint(size=shape2, low=0, high=2)
shape3 = (21, 13)
X_test = np.random.randn(*shape3)


run(X, Y, X_test)

