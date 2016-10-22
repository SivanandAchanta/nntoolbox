# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 22:55:02 2016

% Purpose : Simple Recurrent Layer

@author: Sivanand Achanta
"""

import numpy as np
import actfn


class SimpleRecurrentLayer:

    def get_params(self):
        self.params = []
        self.params.append(self.Wi)
        self.params.append(self.Wfr)
        self.params.append(self.bh)

        return self.params

    def do_init_gparams(self):
        init_gparams = []

        for i in range(len(self.params)):
            init_gparams.append(np.zeros(self.params[i].shape))

        return init_gparams

    def __init__(self, dh, din, init_meth, parser):

        si = parser['hyperparams'].getfloat('si')
        ri = parser['hyperparams'].getfloat('ri')

        if init_meth == 'yi':
            self.Wi = si*np.random.rand(dh, din)
            self.Wfr = ri*np.eye(dh)
            self.bh = np.zeros((dh, 1))

        self.num_units = dh
        self.params = self.get_params()
        self.ngparams = self.do_init_gparams()
        self.pgparams = self.do_init_gparams()
        self.acc_gst = self.do_init_gparams()
        self.acc_dxt = self.do_init_gparams()


    def compute_preactivation(self, X, hp):

        self.Pac = X + np.dot(self.params[1], hp) + self.params[2]
        self.Pac = self.Pac.T

        # return(self)


    def fp(self, X, f):

        s = X.shape
        sl = s[0]
        dh = self.num_units

        hp = np.zeros((dh, 1))
        self.Ac = np.zeros((sl, dh))
        WiX = np.dot(self.params[0], X.T)

        for i in range(sl):

            self.compute_preactivation(WiX[:, i].reshape(dh, 1), hp)
            hp = actfn.activation_function(self.Pac, f)

            self.Ac[i, :] = hp.T.reshape(1, dh)
            hp = hp.reshape(dh, 1)

        # return(self)


    def compute_prelayererror(self):

        self.Ep = np.dot(self.E, self.params[0])

        # return(self)


    def compute_errorsignal(self, iE, f):

        Dac = actfn.der_activation_function(self.Ac, f)
        Dac = Dac.T
        iE = iE.T

        s = Dac.shape
        dh = self.num_units
        sl = s[1]

        self.E = np.zeros((sl, dh))
        en = np.zeros((dh, 1))

        self.params[1] = self.params[1].T

        for i in range(sl-1, -1, -1):

            enu = (Dac[:, i]*(np.dot(self.params[1], en).T + iE[:, i])).T

            self.E[i, :] = enu.T
            en = enu

        self.params[1] = self.params[1].T

        # return(self)


    def compute_gradients(self, X):

        dh = self.num_units

        Hp = np.concatenate((np.zeros((1, dh)), self.Ac[0:-1, :]), axis=0)

        gWi = np.dot(self.E.T, X)
        gWfr = np.dot(self.E.T, Hp)
        gbh = np.sum(self.E, axis=0, keepdims=True).T

        self.gparams = []
        self.gparams.append(gWi)
        self.gparams.append(gWfr)
        self.gparams.append(gbh)

        # return(self)


    def bp(self, iE, X, f):

        self.compute_errorsignal(iE, f)
        self.compute_gradients(X)
        self.compute_prelayererror()

        # return(self)


    '''
    def compute_prelayererror(self, E):

        self.Ep = np.dot(E, self.params[0])

        return(self.Ep)


    def compute_gradients(self, iE, Dac, Ac, X):

        iE = iE.T
        Dac = Dac.T

        s = Dac.shape
        dh = self.num_units
        sl = s[1]

        Hp = np.concatenate((np.zeros((1, dh)), Ac[0:-1, :]), axis=0)

        E = np.zeros((sl, dh))
        en = np.zeros((dh, 1))
        self.params[1] = self.params[1].T

        for i in range(sl-1,-1,-1):
            enu = (Dac[:, i]*(np.dot(self.params[1], en).T + iE[:, i])).T
            E[i, :] = enu.T
            en = enu

        self.params[1] = self.params[1].T
        gWi = np.dot(E.T, X)
        gWfr = np.dot(E.T, Hp)
        gbh = np.sum(E, axis=0, keepdims=True).T

        return(gWi, gWfr, gbh, E)


    def bp(self, iE, X, Ac, f):

        Dac = actfn.der_activation_function(Ac, f)

        [gWi, gWfr, gbh, E] = self.compute_gradients(iE, Dac, Ac, X)
        self.Ep = self.compute_prelayererror(E)

        self.gparams = []
        self.gparams.append(gWi)
        self.gparams.append(gWfr)
        self.gparams.append(gbh)

        return(self.gparams, self.Ep)
    '''
