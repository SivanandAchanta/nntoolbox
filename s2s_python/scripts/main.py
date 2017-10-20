#!/usr/bin/python3.5

'''
Purpose: Main function for training End-to-End TTS


Author: Sivanand Achanta

Date V0: 04-09-2017
'''

import argparse
import os
from data import make_prompts_dict, make_vocab, load_targets, phn2id2phn, save_stats
import torch
from torch.autograd import Variable
import numpy as np

import sys
curr_dir = os.getcwd()
sys.path.append(os.path.realpath(curr_dir + '/../models'))
import encoders
import decoders

from torch import optim
import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pylab
import logging

use_cuda = torch.cuda.is_available()

if use_cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def indexesFromSentence(char2ix, sentence):
    return [char2ix[char] for char in sentence]


def variableFromSentence(phn2id, sentence):
    indexes = indexesFromSentence(phn2id, sentence)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


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



def evaluate(encoder, decoder, prompts, targets, seq_len, phn2id, id2phn, vocab_size, use_cuda, criterion, op_dim, teacher_force):

    val_loss = 0
    for j, k in enumerate(prompts):

        [input_variable, input_length] = get_x_1hot(prompts, k, phn2id, vocab_size, use_cuda)
        [target_variable, target_variable2, target_length] = get_y(seq_len, j, targets, use_cuda)

        loss = 0

        encoder_h0 = encoder.initHidden()
        encoder_c0 = encoder.initHidden()
        encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        encoder_output, (encoder_hn, encoder_cn) = encoder(
              input_variable, (encoder_h0, encoder_c0))
        encoder_outputs = encoder_output.squeeze(1)

        decoder_input = Variable(torch.zeros(1, op_dim))  # all - zero frame
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_h1 = decoder.initHidden()
        decoder_h2 = decoder.initHidden()
        decoder_h3 = decoder.initHidden()
        decoder_c1 = decoder.initHidden()
        decoder_c2 = decoder.initHidden()
        decoder_c3 = decoder.initHidden()
        decoder_attentions = torch.zeros(target_length, input_length)

        if teacher_force:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output1, decoder_output2, decoder_h1, decoder_c1, decoder_h2, decoder_c2, decoder_h3, decoder_c3, decoder_attention = decoder(
                    decoder_input, decoder_h1, decoder_c1, decoder_h2, decoder_c2, decoder_h3, decoder_c3, encoder_outputs)
                loss += criterion(decoder_output1, target_variable[di])
                decoder_input = target_variable2[di].unsqueeze(0)  # Teacher forcing
                decoder_attentions[di] = decoder_attention.data
        else:
            # Professor forcing: Feed the "predicted" target as the next input
            for di in range(target_length):
                decoder_output1, decoder_output2, decoder_h1, decoder_c1, decoder_h2, decoder_c2, decoder_h3, decoder_c3, decoder_attention = decoder(
                    decoder_input, decoder_h1, decoder_c1, decoder_h2, decoder_c2, decoder_h3, decoder_c3, encoder_outputs)
                loss += criterion(decoder_output1, target_variable[di])
                decoder_input = decoder_output2  # Prof. forcing
                decoder_attentions[di] = decoder_attention.data

        val_loss += (loss.data[0] / target_length)

    return(val_loss/len(prompts), decoder_attentions)


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


def get_y(seq_len, j, targets, use_cuda):

    # create target variable
    si = seq_len[j]
    ei = seq_len[j+1]

    outputs = targets[si:ei]
    s = outputs.size()

    # take "2" frames together
    oe = outputs[::2]
    oo = outputs[1::2]

    if s[0] % 2 == 0:
        o = torch.cat((oe, oo), 1)
    else:
        oe = oe[:-1]
        o = torch.cat((oe, oo), 1)

    target_variable = Variable(o)
    target_variable = target_variable.cuda() if use_cuda else target_variable

    target_variable2 = Variable(oo)
    target_variable2 = target_variable2.cuda() if use_cuda else target_variable2

    target_length = target_variable.size()[0]

    return(target_variable, target_variable2, target_length)


def fp_encoder(encoder, input_variable, input_length, use_cuda):

    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(input_length, 2*encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
              input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    return(encoder_output, encoder_outputs)


def train(opt):

    '''
    data could be loaded to a dictionary with "train"/"val"/"test" pointers (Need to improve the below part)
    '''

    print_every = opt.print_every
    showatt_every = opt.print_every
    plot_every = opt.print_every

    full_model_name = opt.model_name \
    + '_hs1' + str(opt.hs1) + '_hs2' + str(opt.hs2) \
    + '_lr' + str(opt.lr) + '_b1' + str(opt.b1) + '_b2' + str(opt.b2) \
    + '_dp' + str(opt.dp) \
    + '_gc' + str(opt.gcth) \
    + '_wtinit' + str(opt.wtinit_meth) \
    + '_ef' + str(int(opt.embedding_flag)) \
    + '_rf' + str(int(opt.residual_flag))

    print(full_model_name)

    logging.basicConfig(filename=opt.log_folder + full_model_name + '.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load training data
    phase = 'train'
    train_prompts = make_prompts_dict(opt.etc_dir + phase + '.done.data')
    vocab = make_vocab(train_prompts)
    phn2id, id2phn = phn2id2phn(vocab)
    file_list = train_prompts.keys()
    save_stats(opt.feats_dir + phase + '/audio_feats/',
    file_list, opt.audio_feats_ext, dtype, '../stats/')
    train_targets, train_seq_len = load_targets(opt.feats_dir + phase
                                                + '/audio_feats/', file_list,
                                                opt.audio_feats_ext,
                                                dtype, opt.stats_dir)


    # Load validation data
    phase = 'val'
    val_prompts = make_prompts_dict(opt.etc_dir + phase + '.done.data')
    file_list = val_prompts.keys()
    val_targets, val_seq_len = load_targets(opt.feats_dir + phase
                                            + '/audio_feats/', file_list,
                                            opt.audio_feats_ext,
                                            dtype, opt.stats_dir)

    # Initialize model
    vocab_size = len(vocab)
    op_dim = 60
    encoder = encoders.EncoderBLSTM_WOE(vocab_size, opt.hs1)
    if opt.residual_flag:
        decoder = decoders.AttnDecoderLSTM3L_R2_Rescon(op_dim, opt.hs2, op_dim, 1, opt.dp)
    else:
        decoder = decoders.AttnDecoderLSTM3L_R2(op_dim, opt.hs2, op_dim, 1, opt.dp)

    encoder = encoder.cuda() if use_cuda else encoder
    decoder = decoder.cuda() if use_cuda else decoder
    criterion = torch.nn.L1Loss(size_average=False)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr,
                                   betas=(opt.b1, opt.b2))
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr,
                                   betas=(opt.b1, opt.b2))

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    best_val_loss = 1000000

    for iter in range(1, opt.niter + 1):

        for j, k in enumerate(train_prompts):

            [input_variable, input_length] = get_x_1hot(train_prompts, k,
            phn2id, vocab_size, use_cuda)
            [target_variable, target_variable2, target_length] = get_y(
            train_seq_len, j, train_targets, use_cuda)

            loss = 0
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_h0 = encoder.initHidden()
            encoder_c0 = encoder.initHidden()
            encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size))
            encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

            encoder_output, (encoder_hn, encoder_cn) = encoder(
                  input_variable, (encoder_h0, encoder_c0))
            encoder_outputs = encoder_output.squeeze(1)

            decoder_input = Variable(torch.zeros(1, op_dim))  # all - zero frame
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            decoder_h1 = decoder.initHidden()
            decoder_c1 = decoder.initHidden()
            decoder_h2 = decoder.initHidden()
            decoder_c2 = decoder.initHidden()
            decoder_h3 = decoder.initHidden()
            decoder_c3 = decoder.initHidden()

            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output1, decoder_output2, decoder_h1, decoder_c1, decoder_h2, decoder_c2, decoder_h3, decoder_c3, decoder_attention = decoder(
                    decoder_input, decoder_h1, decoder_c1, decoder_h2, decoder_c2, decoder_h3, decoder_c3, encoder_outputs)
                loss += criterion(decoder_output1, target_variable[di])
                decoder_input = target_variable2[di].unsqueeze(0)  # Teacher forcing

            loss.backward()
            #torch.nn.utils.clip_grad_norm(encoder.parameters(), 1)
            #torch.nn.utils.clip_grad_norm(decoder.parameters(), 1)
            encoder_optimizer.step()
            decoder_optimizer.step()

            print_loss_total += (loss.data[0] / target_length)
            plot_loss_total += (loss.data[0] / target_length)

            if (j+1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, (iter*len(train_prompts)  - len(train_prompts) + j) / ((opt.niter+1)*len(train_prompts))),
                                             iter, iter / opt.niter * 100, print_loss_avg))

                tf = True # teacher forcing
                avg_val_loss_tf, decoder_attentions_tf = evaluate(encoder.eval(), decoder.eval(), val_prompts, val_targets, val_seq_len, phn2id, id2phn, vocab_size, use_cuda, criterion, op_dim, tf)
                print('%d %0.4f' % (iter, avg_val_loss_tf))

                tf = False # professor forcing
                avg_val_loss_pf, decoder_attentions_pf = evaluate(encoder.eval(), decoder.eval(), val_prompts, val_targets, val_seq_len, phn2id, id2phn, vocab_size, use_cuda, criterion, op_dim, tf)
                print('%d %0.4f' % (iter, avg_val_loss_pf))
                logging.debug('Epoch: ' + str(iter) + ' Update: ' + str(iter*len(train_prompts) - len(train_prompts) + j) + ' Avg Val Loss TF: ' + str(avg_val_loss_tf) + ' Avg Val Loss PF: ' + str(avg_val_loss_pf))

                if avg_val_loss_tf < best_val_loss:
                    best_val_loss = avg_val_loss_tf
                    torch.save(encoder.state_dict(), '%s/%s_enc_epoch_%d_%d.pth' %(opt.model_folder, full_model_name, j, iter))
                    torch.save(decoder.state_dict(), '%s/%s_dec_epoch_%d_%d.pth' %(opt.model_folder, full_model_name, j, iter))

                encoder.train()
                decoder.train()

            if (j+1) % showatt_every == 0:

                plt.figure(1, figsize=(12, 12))
                plt.imshow(decoder_attentions_tf.numpy())
                plt.colorbar()
                pylab.savefig(opt.plot_folder + full_model_name + '_' + str(j) + '_' + str(iter) + '.png', bbox_inches='tight')
                plt.close()

                plt.figure(1, figsize=(12, 12))
                plt.imshow(decoder_attentions_pf.numpy())
                plt.colorbar()
                pylab.savefig(opt.plot_folder + full_model_name + '_' + str(j) + '_' + str(iter) + '.png', bbox_inches='tight')
                plt.close()

            if (j+1) % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)


if __name__ == "__main__":

    # parse the arguments
    parser = argparse.ArgumentParser()

    # Input directories (Feature files)
    parser.add_argument('--etc_dir', required=False, type=str, default='../../etc/', help='../etc/')
    parser.add_argument('--feats_dir', required=False, type=str, default='../../feats/', help='../feats/')
    parser.add_argument('--stats_dir', required=False, type=str, default='../../stats/', help='../feats/')
    parser.add_argument('--audio_feats_ext', required=False, type=str, default='.fb', help='.fb | .mfcc')

    # Hyper-Paramters for SGD
    parser.add_argument('--niter', required=False, type=int, default=20, help='10')
    parser.add_argument('--lr', required=False, type=float, default=0.0003, help='0.0003')
    parser.add_argument('--b1', required=False, type=float, default=0.9, help='0.8| 0.9| 0.95')
    parser.add_argument('--b2', required=False, type=float, default=0.99, help='0.99| 0.995| 0.999')

    # Architecture name and hidden layer sizes
    parser.add_argument('--model_name', required=False, type=str, default='s2s_enc_blstm_dec_lstm3l', help='s2s')
    parser.add_argument('--hs1', required=False, type=int, default=250, help='128| 256')
    parser.add_argument('--hs2', required=False, type=int, default=500, help='256| 512')

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

    # Additional flags
    parser.add_argument('--embedding_flag', required=False, type=bool, default=False, help='True | False')
    parser.add_argument('--residual_flag', required=False, type=bool, default=False, help='True | Fasle')
    parser.add_argument('--print_every', required=False, type=int, default=1000, help='200 | 1000')


    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.model_folder)
    except OSError:
        pass

    try:
        os.makedirs(opt.log_folder)
    except OSError:
        pass

    try:
        os.makedirs(opt.plot_folder)
    except OSError:
        pass

    train(opt)


