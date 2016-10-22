# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:46:57 2016


++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Input/Output Description:
Inputs : [1]


Outputs:


@author: Sivanand Achanta
"""

import numpy as np
import actfn
import scipy.io


class FeedForwardLayer:

    def get_params(self):
        self.params = []
        self.params.append(self.W)
        self.params.append(self.b)

        return self.params

    def do_init_gparams(self):
        init_gparams = []

        for i in range(len(self.params)):
            init_gparams.append(np.zeros(self.params[i].shape))

        return init_gparams


    def __init__(self, dh, din, init_meth, parser):

        if init_meth == 'yi':
            maxweight = np.sqrt(6/(dh+din))
            
            self.W = 2*maxweight*np.random.rand(dh, din) - maxweight                        
            self.b = np.zeros((dh, 1))
            
        elif init_meth == 'ri':
            maxweight = np.sqrt(2/din)

            self.W = maxweight*np.random.randn(dh, din)
            self.b = np.zeros((dh, 1))

        elif init_meth == 'ki':
            maxweight = 3/np.sqrt(din)

            self.W = 2*maxweight*np.random.rand(dh, din) - maxweight
            self.b = np.zeros((dh, 1))

        self.l2 = parser['hyperparams'].getfloat('l2')
        self.params = self.get_params()
        self.ngparams = self.do_init_gparams()
        self.pgparams = self.do_init_gparams()
        self.acc_gst = self.do_init_gparams()
        self.acc_dxt = self.do_init_gparams()

    def compute_preactivation(self, X):
        self.Pac = np.dot(self.params[0], X.T) + self.params[1]
        self.Pac = self.Pac.T

        # return(self)

    def fp(self, X, f):

        self.compute_preactivation(X)
        self.Ac = actfn.activation_function(self.Pac, f)

        # return(self)

    def compute_prelayererror(self):
        self.Ep = np.dot(self.E, self.params[0])

        # return(self)

    def compute_gradients(self, X):
        gW = np.dot(self.E.T, X) + self.l2*2*self.params[0] 
        gb = np.sum(self.E, axis=0, keepdims=True).T

        self.gparams = []
        self.gparams.append(gW)
        self.gparams.append(gb)

        # return(self)

    def bp(self, iE, X, f):

        Dac = actfn.der_activation_function(self.Ac, f)
        self.E = iE*Dac

        self.compute_gradients(X)
        self.compute_prelayererror()


        # return(self)



# Debugging code
'''
def test(X, D, f):
    # Forward Propogate
    Ac    = l1.fp(X, f)

    # Compute Loss
    loss = lossfn.l2_loss(D, Ac)

    return loss


def train(num_epochs, train_numbats, path, f):

    num_up = 0

    for i in range(num_epochs):

        for j in range(train_numbats):

            # Read Mini-Batch Data
            [X, D, batch_size]    = readdata.get_xd(path)

            # Forward Propogate
            l1.Ac               = l1.fp(X, f[0])

            # Compute Loss
            loss_train          = lossfn.l2_loss(D, l1.Ac)

            # Print Loss Value
            num_up              = num_up + 1

            print('Epoch: ' + str(i) + ' Update: ' + str(num_up) + 'Loss: ' + str(loss_train))

            # Compute the Gradients
            E                   = oplayererrorsignal.compute_outputlayererror(D, l1.Ac, f, batch_size)
            [l1.gparams, l1.E]   = l1.bp(E, X, l1.Ac, f[0])


            # Update the Parameters
            l1.params           = graddescent.update_params(l1.params, l1.gparams, sgd_meth)

            #print(l1.params)

            # Test the model on Validation Set
            # if np.mod(num_up, check_valfreq):

            #    [X, D]    = get_xd()
            #    loss_val = test(X, D, f)
            #    print('Epoch: ' + i + ' Update: ' + num_up + 'Loss: ' + loss_val)


# Config Variables
path          = '../../pymats/'
num_epochs    = 3
f             = ['lin']
dh            = 3
din           = 2
train_numbats = 1
sgd_meth      = 'sgdcm'
check_valfreq = 5

# Instatiate the class
l1            = FeedForwardLayer(dh,din)

train(num_epochs, train_numbats, path, f)
'''