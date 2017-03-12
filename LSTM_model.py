#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date: 2017-03-05
"""


import theano
import theano.tensor as T
from layers import InputLayer, LSTMLayer, DropoutLayer, FullyConnectedLayer
from methods import make_caches, get_params, SGD, momentum, floatX



class LSTM_Diagnosis:

    def __init__(self, num_input=256, num_hidden=[512,512], num_output=256, clip_at=0.0, scale_norm=0.0):
        X = T.fmatrix()
        Y = T.imatrix()
        lr = T.fscalar()
        alpha = T.fscalar()
        reg = T.fscalar()
        dropout_prob = T.fscalar()

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.clip_at = clip_at
        self.scale_norm = scale_norm

        inputs = InputLayer(X, name='inputs')
        num_prev = num_input
        prev_layer = inputs

        self.layers = [inputs]
        if type(num_hidden) is types.IntType:
            lstm = LSTMLayer(num_prev, num_hidden, input_layers=[prev_layer], name="lstm", go_backwards=False)
            num_prev = num_hidden
            prev_layer = lstm
            self.layers.append(prev_layer)
            prev_layer = DropoutLayer(prev_layer, dropout_prob=dropout_prob)
            self.layers.append(prev_layer)

        else:
            for i, num_curr in enumerate(num_hidden):
                lstm = LSTMLayer(num_prev, num_curr, input_layers=[prev_layer], name="lstm{0}".format(i + 1), go_backwards=False)
                num_prev = num_curr
                prev_layer = lstm
                self.layers.append(prev_layer)
                prev_layer = DropoutLayer(prev_layer, dropout_prob=dropout_prob)
                self.layers.append(prev_layer)


        FC = FullyConnectedLayer(num_prev, num_output, input_layers=[prev_layer], name="yhat")
        self.layers.append(FC)
        Y_hat = FC.output()
	
	# change to probilities
        Y_hat = T.nnet.softmax(Y_hat)

        params = get_params(self.layers)
        caches = make_caches(params)


        mean_loss = -T.mean(Y * T.log(Y_hat) + (1 - Y) * T.log(1 - Y_hat))
        last_step_loss = -T.mean(Y[-1] * T.log(Y_hat[-1]) + (1 - Y[-1]) * T.log(1 - Y_hat[-1]))
        loss = alpha * mean_loss + (1 - alpha) * last_step_loss
        updates, grads = SGD(loss, params, lr, reg)
        self.train_func = theano.function([X, Y, lr, reg, dropout_prob, alpha], loss, updates=updates, allow_input_downcast=True)

        self.predict_func = theano.function([X, dropout_prob], [Y_hat[-1]], allow_input_downcast=True)

        self.predict_sequence_func = theano.function([X, dropout_prob], [Y_hat], allow_input_downcast=True)

    def train(self, X, Y, lr, alpha, reg, dropout_prob):
        return self.train_func(X, Y, lr, alpha, reg, dropout_prob)

    def predict(self, X):
        return self.predict_func(X, 0.0)  # in predict process dropout = 0

    def predict_sequence(self, X):
        return self.predict_sequence_func(X, 0.0)


