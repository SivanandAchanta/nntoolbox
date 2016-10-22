# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 06:47:11 2016

Purpose : To read train data


@author: Sivanand Achanta
"""


from os import listdir
from os.path import isfile, join
import scipy.io
import numpy as np
import compute_stats


def read_data(datastr, matpath, din, dout):

    files = [f for f in listdir(matpath) if isfile(join(matpath, f))]
    cnt = 1
    if datastr == 'train':
        files = ['train1', 'train2', 'train3', 'train4', 'train5', 'train6' ]
    
    for f in files:
        if f[:len(datastr)] == datastr:

            print('Processing ' + f)
            mat = scipy.io.loadmat(matpath + f)
            data = mat['data']
            targets = mat['targets']
            clv = mat['clv']
            
            if cnt == 1:
                batch_data = data[:, 0:din]
                batch_targets = targets[:, 0:dout]
                batch_clv = clv

            else:
                batch_data = np.concatenate((batch_data[:, 0:din], data), axis=0)
                batch_targets = np.concatenate((batch_targets, targets[:, 0:dout]), axis=0)
                batch_clv = np.concatenate((batch_clv, clv), axis=1)

            cnt = cnt + 1            

    batch_clv = np.cumsum(np.concatenate((np.zeros((1, 1), dtype='uint16'), batch_clv), axis=1))

    return(batch_data, batch_targets, batch_clv)


def normalize_iodata(parser, data, targets, mi, si, maxvi, minvi, mo, so, maxvo, minvo):

    # flags
    mvniflag = parser['flags'].getboolean('mvniflag')
    mvnoflag = parser['flags'].getboolean('mvnoflag')
    maxminiflag = parser['flags'].getboolean('maxminiflag')
    maxminoflag = parser['flags'].getboolean('maxminoflag')
    
    s = targets.shape
    
    nmlvec_in = np.concatenate((np.arange(302,339), np.arange(342,347)),axis = 0)
    nmlvec_out = np.arange(0,s[1])
    
    if mvniflag:
        data = compute_stats.normalize_mv(data, mi, si, nmlvec_in)
    elif maxminiflag:
        data = compute_stats.normalize_maxmin(data, maxvi, minvi, nmlvec_in)
           
    if mvnoflag:
        targets = compute_stats.normalize_mv(targets, mo, so, nmlvec_out)
    elif maxminoflag:
        targets = compute_stats.normalize_maxmin(targets, maxvo, minvo, nmlvec_out)
        
    return(data, targets)


def training_data(parser):

    # paths
    matpath = parser['paths'].get('matpath')
    statspath = parser['paths'].get('matpath')

    # ints
    din = parser['ints'].getint('din')
    dout = parser['ints'].getint('dout')

    datastr = 'train'

    [data, targets, clv] = read_data(datastr, matpath, din, dout)

    # check for nan/inf elements in data and targets
    compute_stats.check_finite(data)
    compute_stats.check_finite(targets)

    # compute stats over data and targets
    [mi, si] = compute_stats.compute_meannstd(data)
    [maxvi, minvi] = compute_stats.compute_maxnmin(data)

    [mo, so] = compute_stats.compute_meannstd(targets)
    [maxvo, minvo] = compute_stats.compute_maxnmin(targets)

    # write stats into a file
    np.save(statspath + 'mi.npy', mi)
    np.save(statspath + 'si.npy', si)
    np.save(statspath + 'maxvi.npy', maxvi)
    np.save(statspath + 'minvi.npy', minvi)

    np.save(statspath + 'mo.npy', mo)
    np.save(statspath + 'so.npy', so)
    np.save(statspath + 'maxvo.npy', maxvo)
    np.save(statspath + 'minvo.npy', minvo)

    mi = mi.astype('float32')
    si = si.astype('float32')
    mo = mo.astype('float32')
    so = so.astype('float32')
    
    [data, targets] = normalize_iodata(parser, data, targets, mi, si, maxvi, minvi, mo, so, maxvo, minvo)
    compute_stats.check_finite(data)
    compute_stats.check_finite(targets)

    return(data, targets, clv)


def validation_data(parser):

    # paths
    matpath = parser['paths'].get('matpath')
    statspath = parser['paths'].get('matpath')

    # ints
    din = parser['ints'].getint('din')
    dout = parser['ints'].getint('dout')

    datastr = 'val'

    [data, targets, clv] = read_data(datastr, matpath, din, dout)
    
    # check for nan/inf elements in data and targets pre-normalization
    compute_stats.check_finite(data)
    compute_stats.check_finite(targets)

    # load stats into from files
    mi = np.load(statspath + 'mi.npy')
    si = np.load(statspath + 'si.npy')
    maxvi = np.load(statspath + 'maxvi.npy')
    minvi = np.load(statspath + 'minvi.npy')

    mo = np.load(statspath + 'mo.npy')
    so = np.load(statspath + 'so.npy')
    maxvo = np.load(statspath + 'maxvo.npy')
    minvo = np.load(statspath + 'minvo.npy')

    mi = mi.astype('float32')
    si = si.astype('float32')
    mo = mo.astype('float32')
    so = so.astype('float32')
    
    # normalize the data
    [data, targets] = normalize_iodata(parser, data, targets, mi, si, maxvi, minvi, mo, so, maxvo, minvo)

    # check for nan/inf elements in data and targets post-normalization
    compute_stats.check_finite(data)
    compute_stats.check_finite(targets)

    return(data, targets, clv)


def test_data(parser):

    # paths
    matpath = parser['paths'].get('matpath')
    statspath = parser['paths'].get('matpath')

    # ints
    din = parser['ints'].getint('din')
    dout = parser['ints'].getint('dout')

    datastr = 'test'

    [data, targets, clv] = read_data(datastr, matpath, din, dout)

    # check for nan.inf elements in train data and targets
    compute_stats.check_finite(data)
    compute_stats.check_finite(targets)

    # load stats into from files
    mi = np.load(statspath + 'mi.npy')
    si = np.load(statspath + 'si.npy')
    maxvi = np.load(statspath + 'maxvi.npy')
    minvi = np.load(statspath + 'minvi.npy')

    mo = np.load(statspath + 'mo.npy')
    so = np.load(statspath + 'so.npy')
    maxvo = np.load(statspath + 'maxvo.npy')
    minvo = np.load(statspath + 'minvo.npy')

    mi = mi.astype('float32')
    si = si.astype('float32')
    mo = mo.astype('float32')
    so = so.astype('float32')
    
    [data, targets] = normalize_iodata(parser, data, targets, mi, si,
                        maxvi, minvi, mo, so, maxvo, minvo)

    compute_stats.check_finite(data)
    compute_stats.check_finite(targets)

    return(data, targets, clv)


def get_xd(data, targets, clv, j):

    si = clv[j]
    ei = clv[j + 1]
    X = data[si:ei, :]
    D = targets[si:ei, :]
    sl = ei - si
    
    X = X.astype('float64')
    D = D.astype('float64')
    
    return(X, D, sl)

'''
matpath = '../../../Blizzard_Test/Telugu/matfiles/'

din     = 347;
dout    = 235;

mvniflag    = 1;
maxminiflag = 0;
mvnoflag    = 0;
maxminoflag = 1;
'''
