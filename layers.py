#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date: 2017-03-05
"""


import numpy as np
import theano
import theano.tensor as T
from methods import sigmoid, dropout, floatX, random_weights, zeros




class NNLayer:
    
    def __init__(self, params):
        self.params = params

    def get_params_names(self):
        return 

    def save_model(self):
        return

    def load_model(self):
        return

    def updates(self):
        return

    def reset_state(self):
        return

class InputLayer(NNLayer):
    """
    """
    def __init__(self, X, name=""):
        self.name = name
        self.X = X
        self.params = []

    def get_params(self):
        return self.params

    def output(self, train=False):
        return self.X

class LSTMLayer(NNLayer):

    def __init__(self, num_input, num_hidden, input_layers=None, name="", go_backwards=False):

        self.name = name
        self.num_input = num_input
        self.num_hidden = num_hidden

        if len(input_layers) >= 2:
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()

        self.h0 = theano.shared(floatX(np.zeros(num_hidden)))
        self.s0 = theano.shared(floatX(np.zeros(num_hidden)))

        self.go_backwards = go_backwards

        self.W_gx = random_weights((num_input, num_hidden), name=self.name+"W_gx")
        self.W_ix = random_weights((num_input, num_hidden), name=self.name+"W_ix")
        self.W_fx = random_weights((num_input, num_hidden), name=self.name+"W_fx")
        self.W_ox = random_weights((num_input, num_hidden), name=self.name+"W_ox")

        self.W_gh = random_weights((num_hidden, num_hidden), name=self.name+"W_gh")
        self.W_ih = random_weights((num_hidden, num_hidden), name=self.name+"W_ih")
        self.W_fh = random_weights((num_hidden, num_hidden), name=self.name+"W_fh")
        self.W_oh = random_weights((num_hidden, num_hidden), name=self.name+"W_oh")

        self.b_g = zeros(num_hidden, name=self.name+"b_g")
        self.b_i = zeros(num_hidden, name=self.name+"b_i")
        self.b_f = zeros(num_hidden, name=self.name+"b_f")
        self.b_o = zeros(num_hidden, name=self.name+"b_o")

        self.params = [self.W_gx, self.W_ix, self.W_ox, self.W_fx,
                       self.W_gh, self.W_ih, self.W_oh, self.W_fh,
                       self.b_g, self.b_i, self.b_f, self.b_o,
                       ]

        self.output()

    def get_params(self):
        return self.params

    def one_step(self, x, h_tm1, s_tm1):
        """
        Run the forward pass for a single timestep of a LSTM
        h_tm1: initial h
        s_tm1: initial s  (cell state)
        """

        g = T.tanh(T.dot(x, self.W_gx) + T.dot(h_tm1, self.W_gh) + self.b_g)
        i = T.nnet.sigmoid(T.dot(x, self.W_ix) + T.dot(h_tm1, self.W_ih) + self.b_i)
        f = T.nnet.sigmoid(T.dot(x, self.W_fx) + T.dot(h_tm1, self.W_fh) + self.b_f)
        o = T.nnet.sigmoid(T.dot(x, self.W_ox) + T.dot(h_tm1, self.W_oh) + self.b_o)

        s = i * g + s_tm1 * f
        h = T.tanh(s) * o

        return h, s


    def output(self, train=True):

        outputs_info = [self.h0, self.s0]

        ([outputs, states], updates) = theano.scan(
            fn=self.one_step,  
            sequences=self.X,
            n_steps=9,
            outputs_info = outputs_info,
            go_backwards=self.go_backwards
        )

        return outputs


    def reset_state(self):
        self.h0 = theano.shared(floatX(np.zeros(self.num_hidden)))
        self.s0 = theano.shared(floatX(np.zeros(self.num_hidden)))

        
        
class GRULayer(NNLayer):
    def __init__(self, num_input, num_cells, input_layers=None, name="", go_backwards=False):
        """
        GRU Layer
        Takes as input sequence of inputs, returns sequence of outputs
        """

        self.name = name
        self.num_input = num_input
        self.num_cells = num_cells

        if len(input_layers) >= 2:
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()

        self.s0 = zeros(num_cells)
        self.go_backwards = go_backwards

        self.U_z = random_weights((num_input, num_cells), name=self.name + "U_z")
        self.W_z = random_weights((num_cells, num_cells), name=self.name + "W_z")
        self.U_r = random_weights((num_input, num_cells), name=self.name + "U_r")
        self.W_r = random_weights((num_cells, num_cells), name=self.name + "W_r")
        self.U_h = random_weights((num_input, num_cells), name=self.name + "U_h")
        self.W_h = random_weights((num_cells, num_cells), name=self.name + "W_h")
        self.b_z = zeros(num_cells, name=self.name + "b_z")
        self.b_r = zeros(num_cells, name=self.name + "b_r")
        self.b_h = zeros(num_cells, name=self.name + "b_h")

        self.params = [self.U_z, self.W_z, self.U_r,
                       self.W_r, self.U_h, self.W_h,
                       self.b_z, self.b_r, self.b_h
                       ]

        self.output()

    def get_params(self):
        return self.params


    def one_step(self, x, s_tm1):
        """
        """
        z = T.nnet.sigmoid(T.dot(x, self.U_z) + T.dot(s_tm1, self.W_z) + self.b_z)
        r = T.nnet.sigmoid(T.dot(x, self.U_r) + T.dot(s_tm1, self.W_r) + self.b_r)
        h = T.tanh(T.dot(x, self.U_h) + T.dot(s_tm1 * r, self.W_h) + self.b_h)
        s = (1 - z) * s_tm1 + z * h

        return [s]

    def output(self, train=True):

        outputs_info = [self.s0]

        (outputs, updates) = theano.scan(
            fn=self.one_step,
            sequences=self.X,
            outputs_info=outputs_info,
            go_backwards=self.go_backwards
        )

        return outputs

    def reset_state(self):
        self.s0 = zeros(self.num_cells)
        

class FullyConnectedLayer(NNLayer):
    """
    """
    def __init__(self, num_input, num_output, input_layers, name=""):

        if len(input_layers) >= 2:
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()
        self.W_yh = random_weights((num_input, num_output),name="W_yh_FC")
        self.b_y = zeros(num_output, name="b_y_FC")
        self.params = [self.W_yh, self.b_y]

    def output(self):
        return T.dot(self.X, self.W_yh) + self.b_y

    def get_params(self):
        return self.params

class SigmoidLayer(NNLayer):
    """
    """
    def __init__(self, num_input, num_output, input_layers, name=""):

        if len(input_layers) >= 2:
            print "number of input layers: %s" % len(input_layers)
            print "len of list comprehension: %s" % len([input_layer.output() for input_layer in input_layers])
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()
        self.W_yh = random_weights((num_input, num_output), name="W_yh")
        self.b_y = zeros(num_output, name="b_y")
        self.params = [self.W_yh, self.b_y]

    def get_params(self):
        return self.params

    def output(self):
        return sigmoid(T.dot(self.X, self.W_yh) + self.b_y)


class SoftmaxLayer(NNLayer):
    """
    """
    def __init__(self, num_input, num_output, input_layer, temperature=1.0, name="FC"):
        self.name = name
        self.X = input_layer
        self.params = []
        self.temp = temperature

        self.W_yh = random_weights((num_input, num_output), name="W_yh")
        self.b_y = zeros(num_output, name="b_y")

        self.params = [self.W_yh, self.b_y]

    def get_params(self):
        return self.params

    def output(self, train=True):
        if train:
            input_sequence = self.X.output(train=True)
        else:
            input_sequence = self.X.output(train=False)

        return softmax((T.dot(input_sequence, self.W_yh) + self.b_y), temperature=self.temp)

class DropoutLayer(NNLayer):

    def __init__(self, input_layer, name="dropout", dropout_prob=0.5):
        self.X = input_layer.output()
        self.params = []
        self.dropout_prob = dropout_prob

    def get_params(self):
        return self.params

    def output(self):
        return dropout(self.X, self.dropout_prob)


class BN(object):

    def __init__(self, input_size, time_steps, momentum=0.1, epsilon=1e-6):
        self.gamma = theano.shared(np.ones(input_size, dtype=np.float32))
        self.beta = theano.shared(np.zeros(input_size, dtype=np.float32))
        self.params = [self.gamma, self.beta]

        self.epsilon = epsilon
        self.momentum = momentum
        self.shared_state = False
        self.train = True
        if not hasattr(BN, 'self.running_mean'):
            self.running_mean = theano.shared(np.zeros((time_steps, input_size), theano.config.floatX))

        if hasattr(BN, 'self.params'):
           print 'you'

        if not hasattr(BN, 'self.running_std'):
            self.running_std = theano.shared(np.zeros((time_steps, input_size), theano.config.floatX))

    def __call__(self, x):
        # batch statistics
        m = x.mean(axis=0)
        std = T.mean((x - m) ** 2 + self.epsilon, axis=0) ** 0.5

        # update shared running averages
        mean_update = self.momentum * self.running_mean + (1-self.momentum) * m
        std_update = self.momentum * self.running_std + (1-self.momentum) * std
        self.updates = [(self.running_mean, mean_update), (self.running_std, std_update)]

        # normalize using running averages
        # (is this better than batch statistics?)
        # (this version seems like it is using the running average
        #  of the previous batch since updates happens after)
        if self.train:
            x = (x - m) / (std + self.epsilon)
        else:
            x = (x - mean_update) / (std_update + self.epsilon)

        # scale and shift
        return self.gamma * x + self.beta

    def set_state(self, input_size, time_steps):
        self.running_mean = theano.shared(np.zeros((time_steps, input_size), theano.config.floatX))
        self.running_std = theano.shared(np.zeros((time_steps, input_size), theano.config.floatX))
        self.shared_state = True

        
 

