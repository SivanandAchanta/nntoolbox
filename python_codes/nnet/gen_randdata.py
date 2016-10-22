#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 22:16:10 2016

@author: Sivanand Achanta
"""

import numpy as np


def gen_randdata(parser):

    # ints
    din = parser['ints'].getint('din')
    dout = parser['ints'].getint('dout')
    N = 50*np.ones((1, 1), dtype='uint16')

    for i in range(10):

        data = np.random.randn(N[0][0], din)

        if parser['strs'].get('oplayer_fn') == 'smax':
            targets = np.zeros((N[0][0], dout))

            for j in range(N[0][0]):
                rix = np.random.randint(dout)
                targets[j, rix] = 1

            # print(targets)
        else:
            targets = np.random.randn(N[0][0], dout)



        if i == 0:
            batch_data = data[:, 0:din]
            batch_targets = targets[:, 0:dout]
            batch_clv = N
        else:

            batch_data = np.concatenate((batch_data[:, 0:din], data),
                                        axis=0)

            batch_targets = np.concatenate((batch_targets,
                                            targets[:, 0:dout]), axis=0)

            batch_clv = np.concatenate((batch_clv, N), axis=1)

    batch_clv = np.cumsum(np.concatenate((np.zeros((1, 1), dtype='uint16'),
                                          batch_clv), axis=1))

    return(batch_data, batch_targets, batch_clv)
