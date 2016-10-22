# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:40:13 2016

@author: Sivanand Achanta
"""

import gen_randdata
import read_data
import feedforwardlayer
import train_model
import gc
import scipy.io
import numpy as np
from configparser import ConfigParser


def main():
    
    """ main """
    parser = ConfigParser()
    parser.read('config_spss.ini')

    
    # read train data
    [train_data,train_targets,train_clv] = read_data.training_data(parser)    
    # read val data
    [val_data,val_targets,val_clv] = read_data.validation_data(parser)
    # read test data
    [test_data,test_targets,test_clv] = read_data.test_data(parser)
    
    '''
    mat = scipy.io.loadmat('../dnn_seqver/train.mat')
    train_data = mat['train_batchdata']
    train_targets = mat['train_batchtargets']
    train_clv = mat['train_clv']
    train_clv = train_clv - 1
    print(train_clv[0])
    print(train_data.shape)
    print(len(train_data))
    
    mat = scipy.io.loadmat('../dnn_seqver/val.mat')
    val_data = mat['val_batchdata']
    val_targets = mat['val_batchtargets']
    val_clv = mat['val_clv']
    val_clv = val_clv - 1
    
    mat = scipy.io.loadmat('../dnn_seqver/test.mat')
    test_data = mat['test_batchdata']
    test_targets = mat['test_batchtargets']
    test_clv = mat['test_clv']
    test_clv = test_clv - 1
    '''
    
    '''
    # read train data
    [train_data, train_targets, train_clv] = gen_randdata.gen_randdata(parser)
    # print(train_targets)
    # read val data
    [val_data, val_targets, val_clv] = gen_randdata.gen_randdata(parser)
    # read test data
    [test_data, test_targets, test_clv] = gen_randdata.gen_randdata(parser)
    '''
    
    train_data_list = [train_data, train_targets, train_clv]
    val_data_list = [val_data, val_targets, val_clv]
    test_data_list = [test_data, test_targets, test_clv]
    
    # Train DNN

    # Config Variables
    arch_name = parser['strs'].get('arch_name')
    init_meth = parser['strs'].get('init_meth')
    olfn = parser['strs'].get('oplayer_fn')
    f = ['relu', 'relu', olfn]

    # Ints
    din = parser['ints'].getint('din')
    dh1 = parser['ints'].getint('dh1')
    dh2 = parser['ints'].getint('dh2')
    dh3 = parser['ints'].getint('dh3')
    dout = parser['ints'].getint('dout')

    # Instatiate the class
    l1 = feedforwardlayer.FeedForwardLayer(dh1, din, init_meth, parser)
    l2 = feedforwardlayer.FeedForwardLayer(dh2, dh1, init_meth, parser)
    #l3 = feedforwardlayer.FeedForwardLayer(dh3, dh2, init_meth, parser)
    l4 = feedforwardlayer.FeedForwardLayer(dout, dh3, init_meth,  parser)
    l = [l1, l2, l4]
    np.save('inital_model_' + arch_name + '.npy',l)
    
    if len(f) != len(l):
        print('Number of non-linearities in f must equal no of layers .. !!!')
        return
    
    # Call train_mlp function
    train_model.train(train_data_list, val_data_list, test_data_list, l, f,
                    parser)

    # Garbage collect
    gc.collect()

if __name__ == "__main__":
    main()
