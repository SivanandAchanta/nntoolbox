# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 14:21:34 2016

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Purpose:

To compute the error signal at output layer

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Description:

[1] Given a loss function (see lossfn.py) we can compute error signal at the
 output layer that needs to be back-propagated through the network.
[2] Mathematically we compute E = (d(J_loss)/d(O))

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Modulewise Description:

Mod1:
    compute_outputlayererror: Given the Loss Function and Output Layer
    Activation function this module computes Error signal at output layer

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Input/Output Description:

Inputs  : D             - Desired output (N x dout)
          O             - Network output (N x dout)
          f             - Activation function at the output layer
          num_samples   - Number of samples in mini-batch (N)

Outputs : E             - Error signal (N x dout)

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Future Work Description:

Integrate more tightly with loss function (lossfn.py) and compute the output
layer error with various loss functions

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@author: Sivanand Achanta
"""


def compute_outputlayererror(D, O, f, num_samples):

    if f[-1] == 'lin':
        E = (O - D)/num_samples

    elif f[-1] == 'smax':
        E = (O - D)/num_samples

    elif f[-1] == 'sigm':
        E = (O - D)/num_samples

    elif f[-1] == 'exp':
        E = (O - D)/num_samples

    return E
