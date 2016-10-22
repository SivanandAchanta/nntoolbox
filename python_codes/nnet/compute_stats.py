# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:32:21 2016

@author: Sivanand Achanta
"""

import numpy as np

verbose_flag = 0


def compute_meannstd(data):
    
    dim_vec = data.shape
    m = np.zeros((1, dim_vec[1]))
    s = np.zeros((1, dim_vec[1]))

    for i in range(dim_vec[1]):
        # compute mean and standard deviation
        m[:, i] = np.mean(data[:, i])
        s[:, i] = np.std(data[:, i])        
        
        if s[:, i] == 0:
            print('STAT CHECK (STD 0): Dimension ', str(i+1), 'has 0 standard deviation !!!')

    if verbose_flag:
        print('-------------------------------------------------------')
        print('--------------- Statistics of the data ----------------')
        print('-------------------------------------------------------')

        print('--------------- Mean of data --------------------------')
        print(m)

        print('--------------- Standard deviation of data ------------')
        print(s)

        print('-------------------------------------------------------')
        print('-------------------------------------------------------')

    s = s + 1e-6
    
    return(m, s)


def compute_maxnmin(data):
    
    dim_vec = data.shape
    maxv = np.zeros((1, dim_vec[1]))
    minv = np.zeros((1, dim_vec[1]))

    for i in range(dim_vec[1]):
        # compute max and min of data
        maxv[:, i] = np.max(data[:, i])
        minv[:, i] = np.min(data[:, i])

    if verbose_flag:
        print('-------------------------------------------------------')
        print('--------------- Statistics of the data ----------------')
        print('-------------------------------------------------------')

        print('--------------- Max of data --------------------------')
        print(maxv)

        print('--------------- Min of data ------------')
        print(minv)

        print('-------------------------------------------------------')
        print('-------------------------------------------------------')

    return(maxv, minv)


def normalize_mv(data, m, s, nmlvec):
    
    for i in nmlvec:
        data[:, i] = data[:, i] - m[:, i]
        data[:, i] = data[:, i]/(s[:, i])

    return(data)


def normalize_maxmin(data, maxv, minv, nmlvec):    

    for i in nmlvec:
        data[:, i] = data[:, i] - minv[:, i]
        data[:, i] = data[:, i]/(maxv[:, i])

    return(data)


def check_isinf(data):

    if np.sum(np.sum(np.isinf(data), axis=1), axis=0) > 0:
        print('There are Inf elements in data')
        return True
    else:
        return False


def check_isnan(data):

    if np.sum(np.sum(np.isnan(data), axis=1), axis=0) > 0:
        print('There are Nan elements in data')
        return True
    else:
        return False


def check_finite(data):

    nan_flag = check_isnan(data)
    inf_flag = check_isinf(data)

    if nan_flag | inf_flag:
        print('There are non-finite elements in data !!!')
    else:
        print('All the elements in data are finite !!!')
