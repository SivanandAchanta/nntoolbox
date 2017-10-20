#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 23:08:34 2017

@author: Sivanand Achanta
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from trasformer import Transformer

use_cuda = torch.cuda.is_available()


######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

class EncoderTF(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_heads, n_layers=1, dropout_p=0.5):
        super(EncoderTF, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.dropout_p = dropout_p
        self.num_heads = num_heads
  
        # Embedding Layer 
        self.embedding = nn.Embedding(self.input_size, self.hidden_size1)
        
        # Transformer Encoder Layer
        self.tf = Transformer(self.hidden_size1, self.hidden_size2, self.num_heads, self.n_layers) 
               

    def forward(self, input, hidden):
        
        # Embedding Layer
        embedded = self.embedding(input.transpose(0,1)).squeeze(0)
        
        # Transformer Encoder
        output = self.tf(embedded)

        return(output) 



class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.pn1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.pn2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.attn1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attn2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(self.hidden_size, 1, bias=False)
        self.smax1 = nn.Softmax()
        self.lstm1 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.lstm2 = nn.LSTM(self.hidden_size*2, self.hidden_size)
        self.lstm3 = nn.LSTM(self.hidden_size*2, self.hidden_size)
        self.linear1 = nn.Linear(self.hidden_size, self.output_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)
        #self.out = nn.Softmax()

        '''
        # Weight Initilization
        me = 0.0
        se = 0.01

        for m in self.modules():
            if isinstance(m, nn.LSTM):

                m.weight_ih_l0.data.normal_(me, se)
                m.bias_ih_l0.data.fill_(0)
                m.weight_hh_l0.data.normal_(me, se)
                m.bias_hh_l0.data.fill_(0)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(me, se)

        self.pn1.bias.data.fill_(0)
        self.pn2.bias.data.fill_(0)
        self.linear.bias.data.fill_(0)
        '''

    def forward(self, input, h1, c1, h2, c2, h3, c3, encoder_outputs):

        # Prenet Layer - 1
        po1 = self.pn1(input)
        po2 = self.relu(po1)
        po3 = self.dropout1(po2)

        # Prenet Layer - 2
        po4 = self.pn2(po3)
        po5 = self.relu(po4)
        po6 = self.dropout2(po5)

        # LSTM - 1
        o1, (h1, c1) = self.lstm1(po6.unsqueeze(1), (h1, c1))
        o2 = o1.squeeze(1)

        # Attention Layer
        way = self.attn1(o2)
        uah = self.attn2(encoder_outputs)
        beta = self.tanh(way.expand_as(uah) + uah)
        gamma = self.attn3(beta)
        attn_weights = self.smax1(gamma.transpose(0,1))
        #print(attn_weights)
        #print(attn_weights.size())
        #print(encoder_outputs.size())
        #print(attn_weights.sum(1))
        attn_applied = torch.mm(attn_weights,
                                 encoder_outputs)

        # LSTM - 2
        o3 = torch.cat((o2, attn_applied), 1)
        o4 = o3.unsqueeze(1)
        o5, (h2, c2) = self.lstm2(o4, (h2, c2))

        # LSTM - 3
        o61 = torch.cat((o5.squeeze(1), attn_applied), 1)
        o6 = o61 + o3 # residual connection
        o7 = o6.unsqueeze(1)
        o8, (h3, c3) = self.lstm3(o7, (h3, c3))
        o9 = o8.squeeze(1)

        # Output Layer
        o10 = self.linear1(o9)
        o11 = self.linear2(o9)
        output = torch.cat((o10, o11), 1)

        return output, o11, h1, c1, h2, c2, h3, c3, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


