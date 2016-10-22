# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:18:46 2016

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Purpose:

To perform gradient update to parameters using various sgd algorithms

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Module Wise Description:

Mod1:
    update_params - updates the parameters of neural network using a
    stochastic-gradient method (W = W - lr*gW)

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Input/Output Description:

Inputs  : [1] params    - list of all paramters of the model
          [2] gparams   - list of all gradients of parameters of the model
          [3] sgd_type  - type of sgd (sgdcm/adam/ada-delta/ada-grad/rms-prop)


Outputs : [1] params    - updated parameters of the model in a list

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Future Work Description:

Implement more advanced/latest sgd types (ada-grad, ada-delta,
rms-prop, adam, adamax)

@author: Sivanand Achanta
"""

import numpy as np


def sgdcm(l, lr, mf):

    for i in range(len(l.params)):
        l.pgparams[i] = mf*l.pgparams[i] - lr*l.gparams[i]
        l.params[i] = l.params[i] + l.pgparams[i]

    #return(params, pgparams)


def adadelta(l, rho, eps_hp, mf):

    for i in range(len(l.params)):
        l.acc_gst[i] = rho*l.acc_gst[i] + (1-rho)*(l.gparams[i]**2)
        l.pgparams[i] = -(np.sqrt(l.acc_dxt[i]+eps_hp)/np.sqrt(l.acc_gst[i]+eps_hp))*l.gparams[i]
        l.acc_dxt[i] = rho*l.acc_dxt[i] + (1-rho)*(l.pgparams[i]**2)

        l.params[i] = l.params[i] + l.pgparams[i]

    #return(params, acc_gst, acc_dxt, pgparams)


def adam(params, pgparams, gparams, alpha, beta1, beta2, lam):

    for i in range(len(params)):
        pgparams[i] = alpha*pgparams[i] + beta1*gparams[i]
        params[i] = params[i] - pgparams[i]

    return(params, pgparams)
