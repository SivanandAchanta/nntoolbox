# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:30:55 2016

Purpose:

To implement various activation functions along with their derivatives

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Description:

Activation functions are applied at every layer in neural networks. These are
usually differentable everywhere.

During backpropagation, the error signal is passed through the derivative of
the activation function.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Module Wise Description:

Mod1:
    activation_function     - computes the activation function given the
    pre-activation values and the type of function


Mod2:
    der_activation_function - computes the derivative of activation function
    given activation function values

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Input/Output Description:

Input and Output description is given on per module basis (see below
 each module)

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Future Work Description:

Parametrize the activation functions (probably in a different module
parametricactfn.py ??)


@author: Sivanand Achanta
"""

import numpy as np

a_tanh = 1.7159
b_tanh = 2/3
bby2a = b_tanh/(2*a_tanh)


def activation_function(pac, f):

    '''
    Inputs  : pac        - Pre-activation Function (N x d)
              f          - Activation Function Name

    Outputs : ac         - Activation Function (N x d)
    '''

    if f == 'tanh':
        ac = a_tanh*np.tanh(b_tanh*pac)

    elif f == 'sigm':
        ac = 1/(1+a_tanh*np.exp(-b_tanh*pac))

    elif f == 'relu':
        ac = np.max((np.zeros(pac.shape), pac), axis=0)

    elif f == 'elu':
        ac = np.max((np.zeros(pac.shape), pac), axis=0)
        ac[pac <= 0] = np.exp(pac[pac <= 0])-1

    elif f == 'smax':
        ac = np.exp(pac)
        ac = ac/np.sum(ac, axis=1, keepdims=True)

    elif f == 'lin':
        ac = pac

    return(ac)


def der_activation_function(ac, f):

    '''
    Inputs  : ac         - Activation Function (N x d)
              f          - Activation Function Name

    Outputs : dac        - Derivative of Activation Function (N x d)
    '''

    if f == 'tanh':
        dac = 2*bby2a*((a_tanh - ac)*(a_tanh + ac))

    elif f == 'sigm':
        dac = b_tanh*(ac*(1 - ac))

    elif f == 'relu':
        dac = np.ones((ac.shape))*(ac > 0)

    elif f == 'elu':
        dac = np.ones((ac.shape))*(ac > 0)
        dac[ac <= 0] = ac[ac <= 0] + 1

    elif f == 'smax':
        dac = np.ones((ac.shape))    # dont use this #####

    elif f == 'lin':
        dac = np.ones((ac.shape))

    return(dac)


'''
# Unit-Tests to make sure that activation and derivatives are
computed appropriately

din = 2;
Nin = 3;

pac = -1*np.ones((Nin,din))

f   = 'tanh';
ac  = activation_function(pac,f);
dac = der_activation_function(ac,f);

print(ac)
print(dac)

f   = 'sigm';
ac  = activation_function(pac,f);
dac = der_activation_function(ac,f);

print(ac)
print(dac)

f   = 'relu';
ac  = activation_function(pac,f);
dac = der_activation_function(ac,f);

print(ac)
print(dac)

f   = 'elu';
ac  = activation_function(pac,f);
dac = der_activation_function(ac,f);

print(ac)
print(dac)

f  = 'smax';
ac = activation_function(pac,f);
print(ac)


f   = 'lin';
ac  = activation_function(pac,f);
dac = der_activation_function(ac,f);

print(ac)
print(dac)
'''
