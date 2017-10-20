#!/usr/bin/python3.5

'''
Purpose: To create data for Tacotron style end-to-end TTS

I/P:
[1] etc_dir: directory containing train.done.data and val.done.data (prompts)
[2] audio_feats_dir: directory containing acoustic feats for targets

O/P:
[1] data and targets sequences

Author: Sivanand Achanta

Date V0: 04-09-2017

'''

import argparse
import os
import numpy as np
import torch
import compute_stats


def make_prompts_dict(prompts_file):
    '''
    Purpose: given prompts (in the format of festival txt.done.data) make a list of prompts


    '''  
    
    f = open(prompts_file, 'r')
    lines = f.readlines()
    line = [line.split() for line in lines]

    prompts = {}
    for item in line:
        six = item.index('"')
        eix = len(item) - 1 - item[::-1].index('"')

        utt = ' '.join(item[six+1:eix])
        prompts[item[1]] = utt

    return(prompts)


def make_prompts_dict_v2(prompts_file, file_list):
    '''
    Purpose: given prompts (in the format of festival txt.done.data) make a list of prompts


    '''

    f = open(prompts_file, 'r')
    lines = f.readlines()
    line = [line.split() for line in lines]

    prompts = {}
    for item in line:
        six = item.index('"')
        eix = len(item) - 1 - item[::-1].index('"')

        utt = ' '.join(item[six+1:eix])

        if item[1] in file_list:
            prompts[item[1]] = utt

    return(prompts)

def make_vocab(prompts):
    
    c = []
    for k in prompts:
        c = c + list(prompts[k])

    return(sorted(set(c)))

def phn2id2phn(vocab):
    phn2id = {}
    id2phn = {} 

    for i,j in enumerate(vocab):
        phn2id[j] = i
        id2phn[i] = j
 
    return(phn2id, id2phn)



def load_targets(targets_dir, file_list, ext, dtype, mo, so, nml_vec):
   
    seq_len = []
    targets = []
  
    for fname in file_list:
        fb = np.load(targets_dir + fname + ext)
        
        targets.extend(fb)
        seq_len.append(fb.shape[0])


    targets = np.asarray(targets)

    #mo = np.load(statspath + 'mo.npy')
    #so = np.load(statspath + 'so.npy')

    #mo = mo.astype('float32')
    #so = so.astype('float32')

    # normalize the data
    #nml_vec = np.arange(0, targets.shape[1])
    targets = compute_stats.normalize_mv(targets, mo, so, nml_vec)

    # check for nan/inf elements in data and targets post-normalization
    # compute_stats.check_finite(targets)

    targets = torch.from_numpy(targets)
    targets = targets.type(dtype)

    seq_len = np.asarray(seq_len)  
    seq_len = seq_len.reshape(seq_len.shape[0],1).transpose()
    seq_len = np.cumsum(np.concatenate((np.zeros((1, 1), dtype='uint16'), seq_len), axis=1))

    return(targets, seq_len)


def save_stats(targets_dir, file_list, ext, dtype, statspath):

    try:
        os.makedirs(statspath)
    except OSError:
        pass
 
    targets = []

    for fname in file_list:
        fb = np.load(targets_dir + fname + ext)

        targets.extend(fb)


    targets = np.asarray(targets)
    print(targets.shape)

    # check for nan/inf elements in data and targets
    compute_stats.check_finite(targets)

    # compute stats over data and targets
    [mo, so] = compute_stats.compute_meannstd(targets)
    [maxvo, minvo] = compute_stats.compute_maxnmin(targets)

    # write stats into a file
    np.save(statspath + 'mo.npy', mo)
    np.save(statspath + 'so.npy', so)
    np.save(statspath + 'maxvo.npy', maxvo)
    np.save(statspath + 'minvo.npy', minvo)

    
def save_stats_suffstats(targets_dir, file_list, ext, dtype, statspath):

    try:
        os.makedirs(statspath)
    except OSError:
        pass

    N = 0
    for fname in file_list:
        targets = np.load(targets_dir + fname + ext)
        #targets = np.asarray(fb)

        # check for nan/inf elements in data and targets
        compute_stats.check_finite(targets)
        
        if N == 0:
            s = np.sum(targets,0, keepdims=True)
            s2 = np.sum(np.square(targets), 0, keepdims=True)
        else:
            s = s + np.sum(targets, 0, keepdims=True)
            s2 = s2 + np.sum(np.square(targets), 0, keepdims=True)

        N = N + targets.shape[0] 
        

    mo = s/N
    so = np.sqrt((s2 - N*(np.square(mo)))/(N-1))

    # write stats into a file
    np.save(statspath + 'mo.npy', mo)
    np.save(statspath + 'so.npy', so)


if __name__ == "__main__":

    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--etc_dir', required=True, type=str, default='../etc/', help='../etc/')
    parser.add_argument('--feats_dir', required=True, type=str, default='../feats/', help='../feats/')
    parser.add_argument('--phase', required=True, type=str, default='train', help='train | val | test')

    opt = parser.parse_args()
    print(opt)

    # make training list of files and validation list of files 
    prompts = make_filelists(opt)
    file_list = prompts.keys()    

    # make vocabulary from train, val and test set
    vocab = make_vocab(prompts)

    # load the targets
    load_targets(opt, file_list) 
