#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 22:19:52 2016

@author: Sivanand Achanta
"""

import gen_randdata
import read_data
import feedforwardlayer
import lstmlayer
import train_model
import gc
from configparser import ConfigParser


def main():
    """ main """
    parser = ConfigParser()
    parser.read('config_spss.ini')

    '''
    # read train data
    [train_data,train_targets,train_clv] = read_data.training_data(parser)
    # read val data
    [val_data,val_targets,val_clv] = read_data.validation_data(parser)
    # read test data
    [test_data,test_targets,test_clv] = read_data.test_data(parser)
    '''

    # read train data
    [train_data, train_targets, train_clv] = gen_randdata.gen_randdata(parser)
    # print(train_targets)
    # read val data
    [val_data, val_targets, val_clv] = gen_randdata.gen_randdata(parser)
    # read test data
    [test_data, test_targets, test_clv] = gen_randdata.gen_randdata(parser)

    train_data_list = [train_data, train_targets, train_clv]
    val_data_list = [val_data, val_targets, val_clv]
    test_data_list = [test_data, test_targets, test_clv]

    # Train DNN

    # Config Variables
    init_meth = parser['strs'].get('init_meth')
    olfn = parser['strs'].get('oplayer_fn')
    f = ['tanh', 'tanh', olfn]

    # Ints
    din = parser['ints'].getint('din')
    dh1 = parser['ints'].getint('dh1')
    dh2 = parser['ints'].getint('dh2')
    dh3 = parser['ints'].getint('dh3')
    dout = parser['ints'].getint('dout')

    # Instatiate the class
    l1 = lstmlayer.LSTMLayer(dh1, din, init_meth, parser)
    # l2 = srnlayer.SimpleRecurrentLayer(dh2, dh1, init_meth, parser)
    l2 = feedforwardlayer.FeedForwardLayer(dout, dh1, init_meth)
    l = [l1, l2]

    # Call train_mlp function
    train_model.train(train_data_list, val_data_list, test_data_list, l, f,
                    parser)

    # Garbage collect
    gc.collect()

if __name__ == "__main__":
    main()
