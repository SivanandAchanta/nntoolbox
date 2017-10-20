#!/usr/bin/python3.5

'''
Purpose: Main function for training End-to-End TTS


Author: Sivanand Achanta

Date V0: 04-09-2017
'''

import argparse
import os
from data import make_prompts_dict, make_vocab, load_targets, phn2id2phn, save_stats, make_prompts_dict_v2, save_stats_suffstats
import torch
from torch.autograd import Variable
from torch import optim
import numpy as np
import sys
curr_dir = os.getcwd()
sys.path.append(os.path.realpath(curr_dir + '/../models'))
import encoders
import decoders
import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pylab
import logging
import gc
import compute_stats
#import soundfile as sf
#import gl

use_cuda = torch.cuda.is_available()

if use_cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


###################################################### Helper Functions ##################################
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
##########################################################################################################




##################################### Get X and Y (Data and Targets) ####################################
'''
Helper functions to get data and targets of a given sequence or a batch of sequences 
'''
 
def indexesFromSentence(char2ix, sentence):
    return [char2ix[char] for char in sentence]


def variableFromSentence(phn2id, sentence):
    indexes = indexesFromSentence(phn2id, sentence)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def get_x(prompts, k, phn2id, use_cuda):

    # create input variable
    char_seq = list(prompts[k])
    input_length = len(char_seq)

    # convert the char_seq to input_index_seq
    input_variable = variableFromSentence(phn2id, char_seq)
    input_variable = input_variable.cuda() if use_cuda else input_variable

    return(input_variable, input_length)


def get_x_1hot(prompts, k, phn2id, vocab_size, use_cuda):

    # create input variable
    char_seq = list(prompts[k])
    input_length = len(char_seq)

    # convert the char_seq to input_index_seq
    indexes = indexesFromSentence(phn2id, char_seq)

    # convert input_varaible into 1-hot representation
    input_variable = torch.zeros(input_length, vocab_size)

    for i, ix in enumerate(indexes):
        input_variable[i, ix] = 1

    input_variable = Variable(input_variable)
    input_variable = input_variable.cuda() if use_cuda else input_variable

    return(input_variable, input_length)


def get_y(seq_len, j, targets, use_cuda, r):

    # create target variable
    si = seq_len[j]
    ei = seq_len[j+1]

    outputs = targets[si:ei]
    s = outputs.size()
      
    if (s[0] % r) != 0:
       outputs = outputs[:-(s[0] % r)]


    # take "r" frames together      
    o = outputs[::r]  
    for i in range(1, r):
        o1 = outputs[i::r]
        o = torch.cat((o, o1), 1)

    target_variable = Variable(o)
    target_variable = target_variable.cuda() if use_cuda else target_variable

    target_variable2 = Variable(o1)
    target_variable2 = target_variable2.cuda() if use_cuda else target_variable2

    target_length = target_variable.size()[0]

    return(target_variable, target_variable2, target_length)

##############################################################################################################





#################################### Train and Evalaute Routines ##############################################
'''
Training and evaluation functions
'''

def evaluate(encoder, decoder, pfnet, prompts, phn2id, id2phn, vocab_size, use_cuda, criterion, op_dim, teacher_force, opt, mo1, so1, nml_vec1, mo2, so2, nml_vec2):

    val_loss_total = 0
    val_loss = 0
    for k in prompts:
        # print('synthesizing ' + k + '...')
        [input_variable, input_length] = get_x(prompts, k, phn2id, use_cuda)
        targets, seq_len = load_targets(opt.feats_dir + '/fb/', [k], '.npy', dtype, mo1, so1, nml_vec1)
        [target_variable, target_variable2, target_length] = get_y(seq_len, 0, targets, use_cuda, opt.r)
        loss = 0

        encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size2))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        encoder_output = encoder(
              input_variable)
        encoder_outputs = encoder_output

        decoder_input = Variable(torch.zeros(1, op_dim))  # all - zero frame
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_h1 = decoder.initHidden()
        decoder_h2 = decoder.initHidden()
        decoder_h3 = decoder.initHidden()
        decoder_c1 = decoder.initHidden()
        decoder_c2 = decoder.initHidden()
        decoder_c3 = decoder.initHidden()
        decoder_attentions = torch.zeros(target_length, input_length)
        decoder_output_half = Variable(torch.zeros(target_length, opt.r*op_dim)).cuda() if use_cuda else Variable(torch.zeros(target_length, opt.r*op_dim))
        decoder_output_full = Variable(torch.zeros(opt.r*target_length, op_dim)).cuda() if use_cuda else Variable(torch.zeros(opt.r*target_length, op_dim))

        if teacher_force:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output1, decoder_output2, decoder_h1, decoder_c1, decoder_h2, decoder_c2, decoder_h3, decoder_c3, decoder_attention = decoder(
                    decoder_input, decoder_h1, decoder_c1, decoder_h2, decoder_c2, decoder_h3, decoder_c3, encoder_outputs)
                loss += criterion(decoder_output1, target_variable[di])
                decoder_input = target_variable2[di].unsqueeze(0)  # Teacher forcing
                decoder_attentions[di] = decoder_attention.data
                decoder_output_half[di] = decoder_output1
        else:
            # Always Sampling: Feed the "predicted" target as the next input
            for di in range(target_length):
                decoder_output1, decoder_output2, decoder_h1, decoder_c1, decoder_h2, decoder_c2, decoder_h3, decoder_c3, decoder_attention = decoder(
                    decoder_input, decoder_h1, decoder_c1, decoder_h2, decoder_c2, decoder_h3, decoder_c3, encoder_outputs)
                loss += criterion(decoder_output1, target_variable[di])
                decoder_input = decoder_output2  # always sampling 
                decoder_attentions[di] = decoder_attention.data
                decoder_output_half[di] = decoder_output1

        # Start Post-Filtering 
        for i in range(opt.r):
            decoder_output_full[i::opt.r,:] = decoder_output_half[:,i*op_dim:(i+1)*op_dim]

        s1 = opt.r*target_length

        targets_pfnet, seq_len_pfnet = load_targets(opt.feats_dir + '/sp/', [k], '.npy', dtype, mo2, so2, nml_vec2)
        targets_pfnet = Variable(targets_pfnet).cuda() if use_cuda else Variable(targets_pfnet)
        s2 = targets_pfnet.size()[0]

        if (s2 % opt.r) > 0:
            targets_pfnet = targets_pfnet[:-(s2 % opt.r),:]

        pfnet_h0 = pfnet.initHidden()
        pfnet_c0 = pfnet.initHidden()
        pfnet_outputs = Variable(torch.zeros(targets_pfnet.size()[0], pfnet.output_size))
        pfnet_outputs = pfnet_outputs.cuda() if use_cuda else pfnet_outputs

        pfnet_output = pfnet(
              decoder_output_full, (pfnet_h0, pfnet_c0))
        pfnet_outputs = pfnet_output
        loss_pfnet = criterion(pfnet_outputs, targets_pfnet)

        loss_total = loss + loss_pfnet

        val_loss += loss.data[0]/(opt.r*target_length)
        val_loss_total += loss_total.data[0]/(opt.r*target_length)

        if not opt.train_flag:
                
            # synthesize
            # denormalize
            pfnet_outputs = pfnet_outputs.data.numpy()
            
            log_pow_spec = compute_stats.denormalize_mv(pfnet_outputs, mo2, so2, nml_vec2)
            
            pow_spec = np.exp(log_pow_spec)
            spec = np.sqrt(pow_spec).transpose(1, 0)
            spec = spec**(1.2)
            y = gl.griffinlim(spec, n_iter = 50, window = np.hanning, n_fft = 1024, hop_length = 200, verbose = False)        
            y = y / np.max(np.abs(y))
            sf.write( opt.synth_folder + opt.full_model_name + '/' + k + '_tf_' + str(teacher_force) + '.wav', y, 16000)
            
            plt.figure(1, figsize=(12, 12))
            plt.imshow(decoder_attentions.numpy())
            plt.colorbar()
            pylab.savefig(opt.plot_folder + opt.full_model_name + '/' + k + '_tf_' + str(teacher_force) + '.png', bbox_inches='tight')
            plt.close()
        
    return(val_loss_total/len(prompts), val_loss/len(prompts), decoder_attentions)

def test(opt):
    

    full_model_name = opt.model_name \
    + '_hs1' + str(opt.hs1) + '_hs2' + str(opt.hs2) + '_pfnet_hs1' + str(opt.pfnet_hs1) + '_r' + str(opt.r) \
    + '_lr' + str(opt.lr) + '_b1' + str(opt.b1) + '_b2' + str(opt.b2) \
    + '_dp' + str(opt.dp) \
    + '_gc' + str(opt.gcth) \
    + '_wtinit' + str(opt.wtinit_meth) \
    + '_lw' + str(int(opt.load_wts)) \
    + '_ef' + str(int(opt.embedding_flag)) \
    + '_rf' + str(int(opt.residual_flag))

    print(full_model_name)
    opt.full_model_name = full_model_name
    
    try:
        os.makedirs(opt.synth_folder + opt.full_model_name)
        os.makedirs(opt.plot_folder + opt.full_model_name)
    except OSError:
        pass
        
    fid = open(opt.feats_dir + 'train_list.txt')
    train_list = fid.read().splitlines()
    fid.close()

    fid = open(opt.feats_dir + 'val_list.txt')
    val_list = fid.read().splitlines()
    fid.close()

    all_prompts = make_prompts_dict_v2(opt.feats_dir + 'txt.done.data', train_list + val_list)
    vocab = make_vocab(all_prompts)
    phn2id, id2phn = phn2id2phn(vocab)
    print(vocab)

    fid = open(opt.feats_dir + 'test_list.txt')
    val_list = fid.read().splitlines()
    val_list = val_list[:10]
    fid.close()

    # Load stats of mfcc
    mo1 = np.load(opt.stats_dir + 'mo.npy')
    so1 = np.load(opt.stats_dir + 'so.npy')
    mo1 = mo1.astype('float32')
    so1 = so1.astype('float32')
    nml_vec1 = np.arange(0, mo1.shape[1])

    # Load stats of spectrum
    mo2 = np.load(opt.pfnet_stats_dir + 'mo.npy')
    so2 = np.load(opt.pfnet_stats_dir + 'so.npy')
    mo2 = mo2.astype('float32')
    so2 = so2.astype('float32')
    nml_vec2 = np.arange(0, mo2.shape[1])

    # Load validation data
    val_prompts = make_prompts_dict_v2(opt.feats_dir + 'txt.done.data', val_list)

    # Initialize model
    vocab_size = len(vocab)
    op_dim = 60
    encoder = encoders.EncoderTF(vocab_size, opt.hs2, 2048, 8, 1)
    if opt.residual_flag:
        if opt.r == 2:
            decoder = decoders.AttnDecoderLSTM3L_R2_Rescon(op_dim, opt.hs2, op_dim, 1, opt.dp)
        if opt.r == 3:
            decoder = decoders.AttnDecoderLSTM3L_R3_Rescon(op_dim, opt.hs2, op_dim, 1, opt.dp)
        if opt.r == 4:
            decoder = decoders.AttnDecoderLSTM3L_R4_Rescon(op_dim, opt.hs2, op_dim, 1, opt.dp)
        if opt.r == 5:
            decoder = decoders.AttnDecoderLSTM3L_R5_Rescon(op_dim, opt.hs2, op_dim, 1, opt.dp)
    else:
        decoder = decoders.AttnDecoderLSTM3L_R2(op_dim, opt.hs2, op_dim, 1, opt.dp)

    op_dim1 = 513
    pfnet = encoders.EncoderBLSTM_WOE_1L(op_dim, opt.pfnet_hs1, op_dim1)

    encoder = encoder.cuda() if use_cuda else encoder
    decoder = decoder.cuda() if use_cuda else decoder
    pfnet = pfnet.cuda() if use_cuda else pfnet
    criterion = torch.nn.L1Loss(size_average=False)

    load_model_name_pfx = '../../wt/' + opt.full_model_name + '_' 
    load_model_name_sfx = '.pth'

    # load model
    enc_state_dict = torch.load(load_model_name_pfx + 'enc' + load_model_name_sfx, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(enc_state_dict)

    dec_state_dict = torch.load(load_model_name_pfx + 'dec' + load_model_name_sfx, map_location=lambda storage, loc: storage)
    decoder.load_state_dict(dec_state_dict)

    pfnet_state_dict = torch.load(load_model_name_pfx + 'pfnet' + load_model_name_sfx, map_location=lambda storage, loc: storage)
    pfnet.load_state_dict(pfnet_state_dict)
    
    
    tf = True # teacher forcing
    avg_val_loss_tf1, avg_val_loss_tf2, decoder_attentions_tf = evaluate(encoder.eval(), decoder.eval(), pfnet.eval(), val_prompts, phn2id, id2phn, vocab_size, use_cuda, criterion, op_dim, tf, opt, mo1, so1, nml_vec1, mo2, so2, nml_vec2)
    print('%0.4f %0.4f' % (avg_val_loss_tf1, avg_val_loss_tf2))

    tf = False # professor forcing
    avg_val_loss_pf1, avg_val_loss_pf2, decoder_attentions_pf = evaluate(encoder.eval(), decoder.eval(), pfnet.eval(), val_prompts, phn2id, id2phn, vocab_size, use_cuda, criterion, op_dim, tf, opt, mo1, so1, nml_vec1, mo2, so2, nml_vec2)
    print('%0.4f %0.4f' % (avg_val_loss_pf1, avg_val_loss_pf2))
    
def train(opt):

    '''
    data could be loaded to a dictionary with "train"/"val"/"test" pointers (Need to improve the below part)
    '''

    print_every = opt.print_every
    showatt_every = opt.print_every
    plot_every = opt.print_every

    full_model_name = opt.model_name \
    + '_hs1' + str(opt.hs1) + '_hs2' + str(opt.hs2) + '_pfnet_hs1' + str(opt.pfnet_hs1) + '_r' + str(opt.r) \
    + '_lr' + str(opt.lr) + '_b1' + str(opt.b1) + '_b2' + str(opt.b2) \
    + '_dp' + str(opt.dp) \
    + '_gc' + str(opt.gcth) \
    + '_wtinit' + str(opt.wtinit_meth) \
    + '_lw' + str(int(opt.load_wts)) \
    + '_ef' + str(int(opt.embedding_flag)) \
    + '_rf' + str(int(opt.residual_flag))

    print(full_model_name)

    logging.basicConfig(filename=opt.log_folder + full_model_name + '.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    r = opt.r
 
    fid = open(opt.feats_dir + 'train_list.txt')
    train_list = fid.read().splitlines()
    fid.close()

    fid = open(opt.feats_dir + 'val_list.txt')
    val_list = fid.read().splitlines()
    fid.close()

    all_prompts = make_prompts_dict_v2(opt.feats_dir + 'txt.done.data', train_list + val_list)
    vocab = make_vocab(all_prompts)
    print(vocab)

    # Load training data
    train_prompts = make_prompts_dict_v2(opt.feats_dir + 'txt.done.data', train_list)
    phn2id, id2phn = phn2id2phn(vocab)
    file_list = train_prompts.keys()
    print(len(file_list), len(train_list))

    # Load stats of mfcc
    mo1 = np.load(opt.stats_dir + 'mo.npy')
    so1 = np.load(opt.stats_dir + 'so.npy')
    mo1 = mo1.astype('float32')
    so1 = so1.astype('float32')
    nml_vec1 = np.arange(0, mo1.shape[1])

    # Load stats of spectrum
    mo2 = np.load(opt.pfnet_stats_dir + 'mo.npy')
    so2 = np.load(opt.pfnet_stats_dir + 'so.npy')
    mo2 = mo2.astype('float32')
    so2 = so2.astype('float32')
    nml_vec2 = np.arange(0, mo2.shape[1])


    # Load validation data
    val_prompts = make_prompts_dict_v2(opt.feats_dir + 'txt.done.data', val_list)

    # Initialize model
    vocab_size = len(vocab)
    op_dim = 60
    encoder = encoders.EncoderTF(vocab_size, opt.hs2, 2048, 8, 1)
    if opt.residual_flag:
        if opt.r == 2:
            decoder = decoders.AttnDecoderLSTM3L_R2_Rescon(op_dim, opt.hs2, op_dim, 1, opt.dp)
        if opt.r == 3:
            decoder = decoders.AttnDecoderLSTM3L_R3_Rescon(op_dim, opt.hs2, op_dim, 1, opt.dp)
        if opt.r == 4:
            decoder = decoders.AttnDecoderLSTM3L_R4_Rescon(op_dim, opt.hs2, op_dim, 1, opt.dp)
        if opt.r == 5:
            decoder = decoders.AttnDecoderLSTM3L_R5_Rescon(op_dim, opt.hs2, op_dim, 1, opt.dp)
    else:
        decoder = decoders.AttnDecoderLSTM3L_R2(op_dim, opt.hs2, op_dim, 1, opt.dp)

    op_dim1 = 513
    pfnet = encoders.EncoderBLSTM_WOE_1L(op_dim, opt.pfnet_hs1, op_dim1)

    encoder = encoder.cuda() if use_cuda else encoder
    decoder = decoder.cuda() if use_cuda else decoder
    pfnet = pfnet.cuda() if use_cuda else pfnet
    criterion = torch.nn.L1Loss(size_average=False)

    if opt.load_wts:
        load_model_name_pfx = '../../wt/s2s_enc_transformer_dec_lstm3l_pfnet_blstm1L_nopfnetloss__hs1512_hs2512_pfnet_hs1256_r4_lr0.0003_b10.9_b20.99_dp0.5_gc0.0_wtinitdefault_init_lw0_ef1_rf1_'
        load_model_name_sfx = '.pth'

        # load model
        enc_state_dict = torch.load(load_model_name_pfx + 'enc' + load_model_name_sfx, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(enc_state_dict)

        dec_state_dict = torch.load(load_model_name_pfx + 'dec' + load_model_name_sfx, map_location=lambda storage, loc: storage)
        decoder.load_state_dict(dec_state_dict)

        pfnet_state_dict = torch.load(load_model_name_pfx + 'pfnet' + load_model_name_sfx, map_location=lambda storage, loc: storage)
        pfnet.load_state_dict(pfnet_state_dict)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr,
                                   betas=(opt.b1, opt.b2))
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr,
                                   betas=(opt.b1, opt.b2))
    pfnet_optimizer = optim.Adam(pfnet.parameters(), lr=opt.lr,
                                   betas=(opt.b1, opt.b2))


    start = time.time()
    print_loss_total = 0  # Reset every print_every
    best_val_loss = sys.maxsize 

    for iter in range(1, opt.niter + 1):

        if iter > 1:
            opt.lr = opt.lr/2
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr,
                                           betas=(opt.b1, opt.b2))
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr,
                                           betas=(opt.b1, opt.b2))
            pfnet_optimizer = optim.Adam(pfnet.parameters(), lr=opt.lr,
                                           betas=(opt.b1, opt.b2))

        for j, k in enumerate(train_prompts):

            [input_variable, input_length] = get_x(train_prompts, k,
            phn2id, use_cuda)

            train_targets, train_seq_len = load_targets(opt.feats_dir + '/fb/', [k], '.npy', dtype, mo1, so1, nml_vec1)

            [target_variable, target_variable2, target_length] = get_y(
            train_seq_len, 0, train_targets, use_cuda, r)

            loss = 0
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size2))
            encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

            encoder_output = encoder(
                  input_variable)
            encoder_outputs = encoder_output

            decoder_input = Variable(torch.zeros(1, op_dim))  # all - zero frame
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            decoder_h1 = decoder.initHidden()
            decoder_c1 = decoder.initHidden()
            decoder_h2 = decoder.initHidden()
            decoder_c2 = decoder.initHidden()
            decoder_h3 = decoder.initHidden()
            decoder_c3 = decoder.initHidden()
            decoder_output_half = Variable(torch.zeros(target_length, r*op_dim)).cuda() if use_cuda else Variable(torch.zeros(target_length, r*op_dim))
            decoder_output_full = Variable(torch.zeros(r*target_length, op_dim)).cuda() if use_cuda else Variable(torch.zeros(r*target_length, op_dim))

            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output1, decoder_output2, decoder_h1, decoder_c1, decoder_h2, decoder_c2, decoder_h3, decoder_c3, decoder_attention = decoder(decoder_input, decoder_h1, decoder_c1, decoder_h2, decoder_c2, decoder_h3, decoder_c3, encoder_outputs)
                loss += criterion(decoder_output1, target_variable[di])
                decoder_input = target_variable2[di].unsqueeze(0)  # Teacher forcing
                decoder_output_half[di] = decoder_output1

            loss.backward(retain_graph=True)
            encoder_optimizer.step()
            decoder_optimizer.step()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Start Post-Filtering Net
            pfnet_optimizer.zero_grad()
            
            for ix in range(r):
                decoder_output_full[ix::r,:] = decoder_output_half[:,ix*op_dim:(ix+1)*op_dim]
                      
            s1 = r*target_length

            train_targets_pfnet, train_seq_len_pfnet = load_targets(opt.feats_dir + '/sp/', [k], '.npy', dtype, mo2, so2, nml_vec2)
            targets_pfnet = Variable(train_targets_pfnet).cuda() if use_cuda else Variable(train_targets_pfnet)
            s2 = targets_pfnet.size()[0]
            if (s2 % r) > 0:
               targets_pfnet = targets_pfnet[:-(s2 % r),:]

            pfnet_h0 = pfnet.initHidden()
            pfnet_c0 = pfnet.initHidden()
            pfnet_outputs = Variable(torch.zeros(targets_pfnet.size()[0], pfnet.output_size))
            pfnet_outputs = pfnet_outputs.cuda() if use_cuda else pfnet_outputs

            pfnet_output = pfnet(
                  decoder_output_full, (pfnet_h0, pfnet_c0))
            pfnet_outputs = pfnet_output
            loss_pfnet = criterion(pfnet_outputs, targets_pfnet)

            loss_pfnet.backward()
            pfnet_optimizer.step()
            loss_total = loss + loss_pfnet

            print_loss_total += (loss_total.data[0] / r*target_length)

            if (j+1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0

                print('%s (%d %d%%) %.4f' % (timeSince(start, (iter*len(train_prompts)  - len(train_prompts) + j) / ((opt.niter+1)*len(train_prompts))), iter, iter / opt.niter * 100, print_loss_avg))

                tf = True # teacher forcing
                avg_total_val_loss_tf, avg_dec_val_loss_tf, decoder_attentions_tf = evaluate(encoder.eval(), decoder.eval(), pfnet.eval(), val_prompts, phn2id, id2phn, vocab_size, use_cuda, criterion, op_dim, tf, opt, mo1, so1, nml_vec1, mo2, so2, nml_vec2)
                print('%d %0.4f %0.4f' % (iter, avg_total_val_loss_tf, avg_dec_val_loss_tf))

                tf = False # always sampling
                avg_total_val_loss_as, avg_dec_val_loss_as, decoder_attentions_pf = evaluate(encoder.eval(), decoder.eval(), pfnet.eval(), val_prompts, phn2id, id2phn, vocab_size, use_cuda, criterion, op_dim, tf, opt, mo1, so1, nml_vec1, mo2, so2, nml_vec2)
                print('%d %0.4f %0.4f' % (iter, avg_total_val_loss_as, avg_dec_val_loss_as))
                logging.debug('Epoch: ' + str(iter) + ' Update: ' + str(iter*len(train_prompts) - len(train_prompts) + j) + ' Avg Total Val Loss TF: ' + str(avg_total_val_loss_tf)  + ' Avg Total Val Loss AS: ' + str(avg_total_val_loss_as)  + ' Avg Dec Val Loss TF: ' + str(avg_dec_val_loss_tf)  + ' Avg Dec Val Loss AS: ' + str(avg_dec_val_loss_as))

                if avg_total_val_loss_tf < best_val_loss:
                    best_val_loss = avg_total_val_loss_tf
                    torch.save(encoder.state_dict(), '%s/%s_enc.pth' %(opt.model_folder, full_model_name ))
                    torch.save(decoder.state_dict(), '%s/%s_dec.pth' %(opt.model_folder, full_model_name ))
                    torch.save(pfnet.state_dict(), '%s/%s_pfnet.pth' %(opt.model_folder, full_model_name ))

                encoder.train()
                decoder.train()
                pfnet.train()

            # if (j+1) % showatt_every == 0:

            #    plt.figure(1, figsize=(12, 12))
            #    plt.imshow(decoder_attentions_tf.numpy())
            #    plt.colorbar()
            #    pylab.savefig(opt.plot_folder + full_model_name + '_' + str(j) + '_' + str(iter) + '.png', bbox_inches='tight')
            #    plt.close()

            #    plt.figure(1, figsize=(12, 12))
            #    plt.imshow(decoder_attentions_pf.numpy())
            #    plt.colorbar()
            #    pylab.savefig(opt.plot_folder + full_model_name + '_' + str(j) + '_' + str(iter) + '.png', bbox_inches='tight')
            #    plt.close()

            # if (j+1) % plot_every == 0:
            #    plot_loss_avg = plot_loss_total / plot_every
            #    plot_losses.append(plot_loss_avg)
            #    plot_loss_total = 0

        gc.collect()

    # showPlot(plot_losses)


if __name__ == "__main__":

    # parse the arguments
    parser = argparse.ArgumentParser()

    # Input directories (Feature files)
    parser.add_argument('--etc_dir', required=False, type=str, default='../../etc/', help='../etc/')
    parser.add_argument('--feats_dir', required=False, type=str, default='../../feats/', help='../feats/')
    parser.add_argument('--pfnet_stats_dir', required=False, type=str, default='../../pfnet_stats/', help='../feats/')
    parser.add_argument('--stats_dir', required=False, type=str, default='../../stats/', help='../feats/')
    parser.add_argument('--pfnet_audio_feats_ext', required=False, type=str, default='.sp', help='.sp| .fb | .mfcc')
    parser.add_argument('--audio_feats_ext', required=False, type=str, default='.fb', help='.fb | .mfcc')

    # Hyper-Paramters for SGD
    parser.add_argument('--niter', required=False, type=int, default=20, help='10')
    parser.add_argument('--lr', required=False, type=float, default=0.0003, help='0.0003')
    parser.add_argument('--b1', required=False, type=float, default=0.9, help='0.8| 0.9| 0.95')
    parser.add_argument('--b2', required=False, type=float, default=0.99, help='0.99| 0.995| 0.999')

    # Architecture name and hidden layer sizes
    parser.add_argument('--model_name', required=False, type=str, default='s2s_enc_transformer_dec_lstm3l_pfnet_blstm1L_nopfnetloss_', help='s2s')
    parser.add_argument('--hs1', required=False, type=int, default=512, help='128| 256')
    parser.add_argument('--pfnet_hs1', required=False, type=int, default=256, help='128| 256')
    parser.add_argument('--hs2', required=False, type=int, default=512, help='256| 512')
    parser.add_argument('--r', required=False, type=int, default=2, help='2| 3| 4| 5')

    # Gradient clipping threshold
    parser.add_argument('--gcth', required=False, type=float, default=0.0, help='0 | 0.5| 1| 50| 100')

    # Dropout factor
    parser.add_argument('--dp', required=False, type=float, default=0.5, help='0.01| 0.1| 0.5')

    # Initialization method
    parser.add_argument('--wtinit_meth', required=False, type=str, default='default_init', help='gi - Gaussian init')

    # Output Folders (Storing Model files, Log files, Plots)
    parser.add_argument('--model_folder', required=False, type=str, default='../../wt/', help='../wt/')
    parser.add_argument('--log_folder', required=False, type=str, default='../../log/', help='../log/')
    parser.add_argument('--plot_folder', required=False, type=str, default='../../plots/', help='../plots/')
    parser.add_argument('--synth_folder', required=False, type=str, default='../../synth_wav/', help='../plots/')

    # Additional flags
    parser.add_argument('--load_wts', required=False, type=bool, default=False, help='True | False')
    parser.add_argument('--embedding_flag', required=False, type=bool, default=True, help='True | False')
    parser.add_argument('--residual_flag', required=False, type=bool, default=True, help='True | Fasle')
    parser.add_argument('--print_every', required=False, type=int, default=10, help='200 | 1000')
    parser.add_argument('--train_flag', required=False, type=bool, default=False, help='True | False')


    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.model_folder)
        os.makedirs(opt.log_folder)
        os.makedirs(opt.plot_folder)
    except OSError:
        pass

        


    if opt.train_flag:
        	train(opt)
    else:
        try:
            os.makedirs(opt.synth_folder)     
        except OSError:
            pass

        test(opt)
        
   


