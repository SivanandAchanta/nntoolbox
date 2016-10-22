# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 12:45:00 2016

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Purpose:

Implement loss functions here

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Description:

[1] Various loss functions are used to train neural nets. This module
implements some widely used loss functions

[2] Typically a loss function needs targets values and the predicted values
from the neural net to compute the empirical loss

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Modulewise Description:

Mod1:
    l2_loss : This is the "Mean squared loss function" used in regression
    problems


Mod2:
    negative_loglikelihood : Used generally in classification tasks
    (also referred to as cross-entropy)

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Input/Output Description:

Inputs  : D         - Desired output (N x dout)
          O         - Network output (N x dout)


Outputs : J_loss    - Loss (scalar)

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Future Work Description:

Implementing regularization losses (l1 and l2) on the parameters

@author: Sivanand Achanta
"""

import numpy as np


def l2_loss(D, O, num_samples):
    J_loss = 0.5*(np.sum(np.sum((D - O)**2, axis=1), axis=0,
                         dtype=np.float64))/num_samples
    return J_loss


def l2_loss_nml(D, O, num_samples):
    J_loss = np.sum(np.sum((D - O)**2, axis=1)/np.sum((D)**2, axis=1),
                    axis=0)/num_samples
    return J_loss


def negative_loglikelihood(D, O, num_samples):
    J_loss = np.sum(-np.sum(D*np.log(O), axis=1))/num_samples
    return J_loss
