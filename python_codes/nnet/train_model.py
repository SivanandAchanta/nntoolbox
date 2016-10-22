# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 21:26:34 2016

@author: Sivanand Achanta
"""

import read_data
import lossfn
import oplayererrorsignal
import graddescent
import numpy as np
import time


def test(l, X, D, f, num_samples):

    # Forward Prop
    O = forward_prop(l, X, f)

    # Compute Loss
    if f[-1] == 'smax':
        loss = lossfn.negative_loglikelihood(D, O, num_samples)
    else:
        loss = lossfn.l2_loss_nml(D, O, num_samples)

    return loss


def test_forngrad(l, X, D, f, num_samples):

    # Forward Prop
    O = forward_prop(l, X, f)

    # Compute Loss
    if f[-1] == 'smax':
        loss = lossfn.negative_loglikelihood(D, O, num_samples)
    else:
        loss = lossfn.l2_loss(D, O, num_samples)

    return loss


def forward_prop(l, X, f):

    l[0].fp(X, f[0])
    for i in range(1, len(l)):

        # Forward Propogate trhough each layer
        l[i].fp(l[i-1].Ac, f[i])
        # print(l[i].Ac)

    return(l[-1].Ac)


def back_prop(l, E, X, f):
    # print(E)
    # print(np.sum(E, axis=0, keepdims=True).T)

    # Back Prop through each layer
    for i in range(len(l)-1, -1, -1):

        if i == len(l)-1 and i == 0:
            l[i].bp(E, X, f[i])
        elif i == len(l)-1:
            l[i].bp(E, l[i-1].Ac, f[i])
        elif i == 0:
            l[i].bp(l[i+1].Ep, X, f[i])
        else:
            l[i].bp(l[i+1].Ep, l[i-1].Ac, f[i])

    '''
    # Back Prop through each layer
    for i in range(len(l)-1, -1, -1):

        if i == len(l)-1 and i == 0:
            [l[i].gparams, l[i].E] = l[i].bp(E, X, l[i].Ac, f[i])
        elif i == len(l)-1:
            [l[i].gparams, l[i].E] = l[i].bp(E, l[i-1].Ac, l[i].Ac, f[i])
        elif i == 0:
            [l[i].gparams, l[i].E] = l[i].bp(l[i+1].E, X, l[i].Ac, f[i])
        else:
            [l[i].gparams, l[i].E] = l[i].bp(l[i+1].E, l[i-1].Ac, l[i].Ac, f[i])
    '''

def update_params(l, sgd_meth, parser):

    # Update the Parameters
    if sgd_meth == 'sgdcm':
        lr = parser['hyperparams'].getfloat('lr')
        mf = parser['hyperparams'].getfloat('mf')
        for i in range(len(l)):
            #graddescent.sgdcm(l[i].params, l[i].pgparams, l[i].gparams, lr, mf)
            graddescent.sgdcm(l[i], lr, mf)

    elif sgd_meth == 'adadelta':
        rho_hp = parser['hyperparams'].getfloat('rho_hp')
        eps_hp = parser['hyperparams'].getfloat('eps_hp')
        mf = parser['hyperparams'].getfloat('mf')
        for i in range(len(l)):
            #graddescent.adadelta(l[i].params, l[i].acc_gst, l[i].acc_dxt, l[i].pgparams, l[i].gparams, rho_hp, eps_hp, mf)
            graddescent.adadelta(l[i], rho_hp, eps_hp, mf)

    # elif sgd_meth == 'adam':


def compute_ngparams(l, D, X, f, num_samples):

    eps_shift = 0.000001

    for i in range(len(l)-1, -1, -1):
        for j in range(len(l[i].params)-1, -1, -1):
            s = l[i].params[j].shape
            for s0 in range(s[0]):
                for s1 in range(s[1]):
                    l[i].params[j][s0][s1] = l[i].params[j][s0][s1] + eps_shift
                    f_xph = test_forngrad(l, X, D, f, num_samples)
                    l[i].params[j][s0][s1] = l[i].params[j][s0][s1] - 2.0*eps_shift

                    # l[i].params[j][s0][s1] = l[i].params[j][s0][s1] - eps_shift
                    f_xnh = test_forngrad(l, X, D, f, num_samples)
                    l[i].params[j][s0][s1] = l[i].params[j][s0][s1] + eps_shift

                    l[i].ngparams[j][s0][s1] = (f_xph - f_xnh)/(2.0*eps_shift)


def check_grad(l, parser):
    for i in range(len(l)-1, -1, -1):
        for j in range(len(l[i].params)-1, -1, -1):
            print(l[i].gparams[j])
            print(l[i].ngparams[j])
            dg = l[i].gparams[j] - l[i].ngparams[j]
            print(np.abs(dg))
            ndg = np.abs(dg)/np.maximum(np.abs(l[i].gparams[j]), np.abs(l[i].ngparams[j]))
            print(ndg)

            print('Shape: ' + str((ndg.shape)) + '\t' + 'Total elements:' + str(len(ndg.flat)))
            print('No. of Elements with more than 1e-4 difference:' + str(len(ndg[np.where(ndg > 1e-4)])))
            time.sleep(5)


def train(train_data_list, val_data_list, test_data_list, l, f, parser):

    # flags
    gradCheckFlag = parser['flags'].getboolean('gradCheckFlag')

    num_epochs = parser['ints'].getint('epochs')
    sgd_meth = parser['strs'].get('sgd_meth')
    arch_name = parser['strs'].get('arch_name')
    
    check_valfreq = parser['ints'].getint('val_freq')
    best_val_loss = parser['ints'].getint('best_val_loss')

    num_up = 0

    train_data = train_data_list[0]
    train_targets = train_data_list[1]
    train_clv = train_data_list[2]
    train_numbats = len(train_clv) - 1
    
    val_data = val_data_list[0]
    val_targets = val_data_list[1]
    val_clv = val_data_list[2]
    val_numbats = len(val_clv) - 1

    test_data = test_data_list[0]
    test_targets = test_data_list[1]
    test_clv = test_data_list[2]
    test_numbats = len(test_clv) - 1

    for i in range(num_epochs):
        # print(i)
        start_time_epoch = time.time()
        
        train_seq = np.arange(0,train_numbats)
        np.random.shuffle(train_seq)
        
        for j in train_seq:
            #print(j)

            # Read Mini-Batch Data
            [X, D, batch_size] = read_data.get_xd(train_data, train_targets, train_clv, j)
            
            # Forward Prop
            O = forward_prop(l, X, f)
            
            # Compute Loss
            # loss_train = lossfn.l2_loss_nml(D, O, batch_size)

            # Print Loss Value
            num_up = num_up + 1
            # print('Epoch: ' + str(i) + ' Update: ' + str(num_up) + ' Loss: ' + str(loss_train))

            # Compute the Gradients - Back Prop
            E = oplayererrorsignal.compute_outputlayererror(D, O, f, batch_size)
            back_prop(l, E, X, f)
            
            # Check Gradients
            # if gradCheckFlag:
            #    compute_ngparams(l, D, X, f, batch_size)
            #    check_grad(l, parser)
                # break
            #    time.sleep(4)

            # Update Params
            update_params(l, sgd_meth, parser)

            # Test the model on Validation Set
            if np.mod(num_up, check_valfreq) == 0:

                val_loss = np.zeros((val_numbats, 1))

                for li in range(val_numbats):
                    [X,D,batch_size] = read_data.get_xd(val_data,val_targets,val_clv,li)
                    val_loss[li] = test(l, X, D, f, batch_size)
                    
                avg_val_loss = np.mean(val_loss)
                print('Epoch: ' + str(i) + ' Update: ' + str(num_up) + ' Avg Val Loss: ' + str(avg_val_loss))

                if best_val_loss > avg_val_loss:
                    best_val_loss = avg_val_loss

                    test_loss = np.zeros((test_numbats, 1))
                    for li in range(test_numbats):
                        [X, D, batch_size] = read_data.get_xd(test_data, test_targets, test_clv, li)
                        test_loss[li] = test(l, X, D, f, batch_size)

                    avg_test_loss = np.mean(test_loss)
                    print('Epoch: ' + str(i) + ' Update: ' + str(num_up) + ' Avg Test Loss: ' + str(avg_test_loss))

                    # save model parameters here
                    np.save('model_' + arch_name + '.npy',l)
                    
        print('Time taken for Epoch %d' % i, 'is %f seconds ---' % (time.time() - start_time_epoch))
         