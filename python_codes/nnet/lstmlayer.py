#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 17:30:15 2016

@author: Sivanand Achanta
"""

import numpy as np
import actfn


class LSTMLayer:

    def get_params(self):
        self.params = []
        self.params.append(self.Wz)
        self.params.append(self.Wi)
        self.params.append(self.Wf)
        self.params.append(self.Wo)

        self.params.append(self.Rz)
        self.params.append(self.Ri)
        self.params.append(self.Rf)
        self.params.append(self.Ro)

        self.params.append(self.bz)
        self.params.append(self.bi)
        self.params.append(self.bf)
        self.params.append(self.bo)

        self.params.append(self.pi)
        self.params.append(self.pf)
        self.params.append(self.po)

        return self.params

    def do_init_gparams(self):
        init_gparams = []

        for i in range(len(self.params)):
            init_gparams.append(np.zeros(self.params[i].shape))

        return init_gparams

    def __init__(self, dh, din, init_meth, parser):

        si = parser['hyperparams'].getfloat('si')

        if init_meth == 'yi':
            self.Wz = si*np.random.rand(dh, din)
            self.Wi = si*np.random.rand(dh, din)
            self.Wf = si*np.random.rand(dh, din)
            self.Wo = si*np.random.rand(dh, din)

            self.Rz = si*np.random.rand(dh, dh)
            self.Ri = si*np.random.rand(dh, dh)
            self.Rf = si*np.random.rand(dh, dh)
            self.Ro = si*np.random.rand(dh, dh)

            self.bz = np.zeros((dh, 1))
            self.bi = np.zeros((dh, 1))
            self.bf = np.zeros((dh, 1))
            self.bo = np.zeros((dh, 1))

            self.pi = np.zeros((dh, 1))
            self.pf = np.zeros((dh, 1))
            self.po = np.zeros((dh, 1))

        self.num_units = dh
        self.params = self.get_params()
        self.ngparams = self.do_init_gparams()
        self.pgparams = self.do_init_gparams()
        self.acc_gst = self.do_init_gparams()
        self.acc_dxt = self.do_init_gparams()


    def compute_preactivation(self, X, hp, i, j):

        Pac = X + np.dot(self.params[i], hp) + self.params[j]

        return(Pac)


    def fp(self, X, f):

        s = X.shape
        sl = s[0]
        dh = self.num_units

        hp = np.zeros((dh, 1))
        cp = np.zeros((dh, 1))

        self.Za = np.zeros((sl, dh))
        self.Ia = np.zeros((sl, dh))
        self.Fa = np.zeros((sl, dh))
        self.Ca = np.zeros((sl, dh))
        self.Oa = np.zeros((sl, dh))
        self.Ha = np.zeros((sl, dh))
        self.Ac = np.zeros((sl, dh))

        WzX = np.dot(self.params[0], X.T)
        WiX = np.dot(self.params[1], X.T)
        WfX = np.dot(self.params[2], X.T)
        WoX = np.dot(self.params[3], X.T)


        for i in range(sl):

            Pac = self.compute_preactivation(WzX[:, i].reshape(dh, 1), hp, 4, 8)
            zt = actfn.activation_function(Pac, 'tanh')

            Pac = self.compute_preactivation(WiX[:, i].reshape(dh, 1), hp, 5, 9)
            Pac = Pac + self.params[12]*cp
            it = actfn.activation_function(Pac, 'sigm')

            Pac = self.compute_preactivation(WfX[:, i].reshape(dh, 1), hp, 6, 10)
            Pac = Pac + self.params[13]*cp
            ft = actfn.activation_function(Pac, 'sigm')

            ct = zt*it + cp*ft
            Pac = self.compute_preactivation(WoX[:, i].reshape(dh, 1), hp, 7, 11)
            Pac = Pac + self.params[14]*ct
            ot = actfn.activation_function(Pac, 'sigm')

            hct = actfn.activation_function(ct, 'tanh')

            ht = hct*ot

            self.Za[i, :] = zt.T.reshape(1, dh)
            self.Ia[i, :] = it.T.reshape(1, dh)
            self.Fa[i, :] = ft.T.reshape(1, dh)
            self.Ca[i, :] = ct.T.reshape(1, dh)
            self.Oa[i, :] = ot.T.reshape(1, dh)
            self.Ha[i, :] = hct.T.reshape(1, dh)
            self.Ac[i, :] = ht.T.reshape(1, dh)

            hp = ht.reshape(dh, 1)
            cp = ct.reshape(dh, 1)

        # return(self)


    def compute_prelayererror(self):

        self.Ep = np.dot(self.Ze, self.params[0]) + np.dot(self.Ie, self.params[1]) + np.dot(self.Fe, self.params[2]) + np.dot(self.Oe, self.params[3])

        # return(self)


    def compute_errorsignal(self, iE, f):

        iE = iE.T

        s = self.Ac.shape
        dh = self.num_units
        sl = s[0]

        self.Ze = np.zeros((sl, dh))
        self.Ie = np.zeros((sl, dh))
        self.Fe = np.zeros((sl, dh))
        self.Ce = np.zeros((sl, dh))
        self.Oe = np.zeros((sl, dh))
        self.He = np.zeros((sl, dh))

        enz = np.zeros((dh, 1))
        eni = np.zeros((dh, 1))
        enf = np.zeros((dh, 1))
        eno = np.zeros((dh, 1))
        enc = np.zeros((dh, 1))

        self.params[4] = self.params[4].T
        self.params[5] = self.params[5].T
        self.params[6] = self.params[6].T
        self.params[7] = self.params[7].T

        self.Za = self.Za.T
        self.Ia = self.Ia.T
        self.Fa = self.Fa.T
        self.Ca = self.Ca.T
        self.Oa = self.Oa.T
        self.Ha = self.Ha.T

        Dza = actfn.der_activation_function(self.Za, 'tanh')
        Dia = actfn.der_activation_function(self.Ia, 'sigm')
        Dfa = actfn.der_activation_function(self.Fa, 'sigm')
        Doa = actfn.der_activation_function(self.Oa, 'sigm')
        Dha = actfn.der_activation_function(self.Ha, 'tanh')


        Fn = np.concatenate((self.Fa[:, 1:], np.zeros((dh, 1))), axis=1)
        Cp = np.concatenate((np.zeros((dh, 1)), self.Ca[:, 0:-1]), axis=1)

        for i in range(sl-1,-1,-1):

            enuh = (iE[:, i] + np.dot(self.params[4], enz).T + np.dot(self.params[5], eni).T + np.dot(self.params[6], enf).T + np.dot(self.params[7], eno).T).T
            enuo = (enuh*self.Ha[:, i].reshape(dh,1)*Doa[:, i].reshape(dh,1))
            enuc = (enuh*self.Oa[:, i].reshape(dh,1)*Dha[:, i].reshape(dh,1)) + (self.po*enuo) + self.pi*eni + self.pf*enf + (enc*Fn[:, i].reshape(dh,1))

            enuf = (enuc.T*Cp[:, i]*Dfa[:, i]).T
            enui = (enuc.T*self.Za[:, i]*Dia[:, i]).T
            enuz = (enuc.T*self.Ia[:, i]*Dza[:, i]).T

            self.He[i, :] = enuh.T
            self.Oe[i, :] = enuo.T
            self.Ce[i, :] = enuc.T
            self.Fe[i, :] = enuf.T
            self.Ie[i, :] = enui.T
            self.Ze[i, :] = enuz.T

            enz = enuz
            eni = enui
            enf = enuf
            enc = enuc
            eno = enuo

        self.params[4] = self.params[4].T
        self.params[5] = self.params[5].T
        self.params[6] = self.params[6].T
        self.params[7] = self.params[7].T

        self.Za = self.Za.T
        self.Ia = self.Ia.T
        self.Fa = self.Fa.T
        self.Ca = self.Ca.T
        self.Oa = self.Oa.T
        self.Ha = self.Ha.T

        # return(self)


    def compute_gradients(self, X):

        dh = self.num_units
        Hp = np.concatenate((np.zeros((1, dh)), self.Ac[0:-1, :]), axis=0)
        Cp = np.concatenate((np.zeros((1, dh)), self.Ca[0:-1, :]), axis=0)

        gWz = np.dot(self.Ze.T, X)
        gWi = np.dot(self.Ie.T, X)
        gWf = np.dot(self.Fe.T, X)
        gWo = np.dot(self.Oe.T, X)

        gRz = np.dot(self.Ze.T, Hp)
        gRi = np.dot(self.Ie.T, Hp)
        gRf = np.dot(self.Fe.T, Hp)
        gRo = np.dot(self.Oe.T, Hp)

        gbz = np.sum(self.Ze, axis=0, keepdims=True).T
        gbi = np.sum(self.Ie, axis=0, keepdims=True).T
        gbf = np.sum(self.Fe, axis=0, keepdims=True).T
        gbo = np.sum(self.Oe, axis=0, keepdims=True).T

        gpi = np.sum(self.Ie*Cp, axis=0, keepdims=True).T
        gpf = np.sum(self.Fe*Cp, axis=0, keepdims=True).T
        gpo = np.sum(self.Oe*self.Ca, axis=0, keepdims=True).T

# gpi = np.zeros((dh, 1))
# gpf = np.zeros((dh, 1))
# gpo = np.zeros((dh, 1))

        self.gparams = []
        self.gparams.append(gWz)
        self.gparams.append(gWi)
        self.gparams.append(gWf)
        self.gparams.append(gWo)

        self.gparams.append(gRz)
        self.gparams.append(gRi)
        self.gparams.append(gRf)
        self.gparams.append(gRo)

        self.gparams.append(gbz)
        self.gparams.append(gbi)
        self.gparams.append(gbf)
        self.gparams.append(gbo)

        self.gparams.append(gpi)
        self.gparams.append(gpf)
        self.gparams.append(gpo)


    def bp(self, iE, X, f):

        self.compute_errorsignal(iE, f)
        self.compute_gradients(X)
        self.compute_prelayererror()

        # return(self)

