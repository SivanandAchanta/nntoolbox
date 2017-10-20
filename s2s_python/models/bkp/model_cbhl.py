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
from highway import Highway

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

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, n_layers=1, dropout_p=0.5):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.dropout_p = dropout_p
        self.num_hl_layers = 4
  
        # Embedding Layer 
        self.embedding = nn.Embedding(self.input_size, self.hidden_size1)
     
        # Prenet
        self.pn1 = nn.Linear(self.hidden_size1, self.hidden_size1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.pn2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.dropout2 = nn.Dropout(self.dropout_p)

        # Convolutional Bank
        self.cn1 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 1, padding=0)
        self.bn1 = nn.BatchNorm1d(self.hidden_size2)
        self.cn2 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 2, padding=1)
        self.bn2 = nn.BatchNorm1d(self.hidden_size2)
        self.cn3 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 3, padding=2)
        self.bn3 = nn.BatchNorm1d(self.hidden_size2)
        self.cn4 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 4, padding=3)
        self.bn4 = nn.BatchNorm1d(self.hidden_size2)
        self.cn5 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 5, padding=4)
        self.bn5 = nn.BatchNorm1d(self.hidden_size2)
        self.cn6 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 6, padding=5)
        self.bn6 = nn.BatchNorm1d(self.hidden_size2)
        self.cn7 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 7, padding=6)
        self.bn7 = nn.BatchNorm1d(self.hidden_size2)
        self.cn8 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 8, padding=7)
        self.bn8 = nn.BatchNorm1d(self.hidden_size2)
        self.cn9 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 9, padding=8)
        self.bn9 = nn.BatchNorm1d(self.hidden_size2)
        self.cn10 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 10, padding=9)
        self.bn10 = nn.BatchNorm1d(self.hidden_size2)
        self.cn11 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 11, padding=10)
        self.bn11 = nn.BatchNorm1d(self.hidden_size2)
        self.cn12 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 12, padding=11)
        self.bn12 = nn.BatchNorm1d(self.hidden_size2)
        self.cn13 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 13, padding=12)
        self.bn13 = nn.BatchNorm1d(self.hidden_size2)
        self.cn14 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 14, padding=13)
        self.bn14 = nn.BatchNorm1d(self.hidden_size2)
        self.cn15 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 15, padding=14)
        self.bn15 = nn.BatchNorm1d(self.hidden_size2)
        self.cn16 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 16, padding=15)
        self.bn16 = nn.BatchNorm1d(self.hidden_size2)

        # Max-Pool Layer
        self.mp = nn.MaxPool1d(2, stride=1, padding=1)

        # Convolution 1-D Projections
        self.cnp1 = nn.Conv1d(self.hidden_size2*16, self.hidden_size2, 3, padding=2)
        self.bnp1 = nn.BatchNorm1d(self.hidden_size2)
        self.cnp2 = nn.Conv1d(self.hidden_size2, self.hidden_size2, 3, padding=2)
        self.bnp2 = nn.BatchNorm1d(self.hidden_size2)

        # Highway Layers
        self.hw = Highway(self.hidden_size2, self.num_hl_layers, f=torch.nn.functional.relu) 

        # BLSTM   
        self.lstm = nn.LSTM(self.hidden_size2, self.hidden_size2, bidirectional=True)

        '''
        # Weight Initilization
        me = 0.0
        se = 0.01

        self.lstm.weight_ih_l0.data.normal_(me, se)
        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.weight_hh_l0.data.normal_(me, se)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.lstm.weight_ih_l0_reverse.data.normal_(me, se)
        self.lstm.bias_ih_l0_reverse.data.fill_(0)
        self.lstm.weight_hh_l0_reverse.data.normal_(me, se)
        self.lstm.bias_hh_l0_reverse.data.fill_(0)
        '''

    def forward(self, input, hidden):
        
        # Embedding Layer
        embedded = self.embedding(input.transpose(0,1)).squeeze(0)

        # Prenet Layer - 1
        po1 = self.pn1(embedded)
        po2 = self.relu(po1)
        po3 = self.dropout1(po2)

        # Prenet Layer - 2
        po4 = self.pn2(po3)
        po5 = self.relu(po4)
        po6 = self.dropout2(po5)

        # Convolution Bank
        po6 = po6.unsqueeze(0)
        po6 = po6.transpose(1,2)

        ocb1 = self.cn1(po6)        
        ocb1 = self.relu(ocb1)
        ocbbn1 = self.bn1(ocb1)
        
        ocb2 = self.cn2(po6)
        ocb2 = ocb2[:, :, 1:]
        ocb2 = self.relu(ocb2)
        ocbbn2 = self.bn2(ocb2)
        cbo = torch.cat((ocbbn1, ocbbn2), 1)

        ocb3 = self.cn3(po6)
        ocb3 = ocb3[:, :, 2:]
        ocb3 = self.relu(ocb3)
        ocbbn3 = self.bn3(ocb3)
        cbo = torch.cat((cbo, ocbbn3), 1)

        ocb4 = self.cn4(po6)
        ocb4 = ocb4[:, :, 3:]
        ocb4 = self.relu(ocb4)
        ocbbn4 = self.bn4(ocb4)
        cbo = torch.cat((cbo, ocbbn4), 1)

        ocb5 = self.cn5(po6)
        ocb5 = ocb5[:, :, 4:]
        ocb5 = self.relu(ocb5)
        ocbbn5 = self.bn5(ocb5)
        cbo = torch.cat((cbo, ocbbn5), 1)

        ocb6 = self.cn6(po6)
        ocb6 = ocb6[:, :, 5:]
        ocb6 = self.relu(ocb6)
        ocbbn6 = self.bn6(ocb6)
        cbo = torch.cat((cbo, ocbbn6), 1)

        ocb7 = self.cn7(po6)
        ocb7 = ocb7[:, :, 6:]
        ocb7 = self.relu(ocb7)
        ocbbn7 = self.bn7(ocb7)
        cbo = torch.cat((cbo, ocbbn7), 1)

        ocb8 = self.cn8(po6)
        ocb8 = ocb8[:, :, 7:]
        ocb8 = self.relu(ocb8)
        ocbbn8 = self.bn8(ocb8)
        cbo = torch.cat((cbo, ocbbn8), 1)

        ocb9 = self.cn9(po6)
        ocb9 = ocb9[:, :, 8:]
        ocb9 = self.relu(ocb9)
        ocbbn9 = self.bn9(ocb9)
        cbo = torch.cat((cbo, ocbbn9), 1)

        ocb10 = self.cn10(po6)
        ocb10 = ocb10[:, :, 9:]
        ocb10 = self.relu(ocb10)
        ocbbn10 = self.bn10(ocb10)
        cbo = torch.cat((cbo, ocbbn10), 1)

        ocb11 = self.cn11(po6)
        ocb11 = ocb11[:, :, 10:]
        ocb11 = self.relu(ocb11)
        ocbbn11 = self.bn11(ocb11)
        cbo = torch.cat((cbo, ocbbn11), 1)

        ocb12 = self.cn12(po6)
        ocb12 = ocb12[:, :, 11:]
        ocb12 = self.relu(ocb12)
        ocbbn12 = self.bn12(ocb12)
        cbo = torch.cat((cbo, ocbbn12), 1)

        ocb13 = self.cn13(po6)
        ocb13 = ocb13[:, :, 12:]
        ocb13 = self.relu(ocb13)
        ocbbn13 = self.bn13(ocb13)
        cbo = torch.cat((cbo, ocbbn13), 1)

        ocb14 = self.cn14(po6)
        ocb14 = ocb14[:, :, 13:]
        ocb14 = self.relu(ocb14)
        ocbbn14 = self.bn14(ocb14)
        cbo = torch.cat((cbo, ocbbn14), 1)

        ocb15 = self.cn15(po6)
        ocb15 = ocb15[:, :, 14:]
        ocb15 = self.relu(ocb15)
        ocbbn15 = self.bn15(ocb15)
        cbo = torch.cat((cbo, ocbbn15), 1)

        ocb16 = self.cn16(po6)
        ocb16 = ocb16[:, :, 15:]
        ocb16 = self.relu(ocb16)
        ocbbn16 = self.bn16(ocb16)
        cbo = torch.cat((cbo, ocbbn16), 1)

        # Max Pooling
        mpo = self.mp(cbo)
        mpo = mpo[:, :, 1:]

        # Convolution 1-D Projections  
        oc1 = self.cnp1(mpo)
        oc1 = oc1[:, :, 2:]
        oc1 = self.relu(oc1)
        ocbn1 = self.bnp1(oc1)
  
        oc2 = self.cnp2(ocbn1)
        oc2 = oc2[:, :, 2:].contiguous()
        ocbn2 = self.bnp2(oc2)

        # Residual Connection 
        ocbn2 = ocbn2 + po6
        
        # From N X C X L to L X N X C
        ocbn2 = ocbn2.transpose(1,2)
        ocbn2 = ocbn2.transpose(0,1)

        # Highway Layers
        ohl = self.hw(ocbn2.squeeze(1))
        ohl = ohl.unsqueeze(1) 
  
        # BLSTM Layer
        output, (hn, cn) = self.lstm(ohl, hidden)
        return output, (hn, cn)

    def initHidden(self):
        result = Variable(torch.zeros(2, 1, self.hidden_size2))
        if use_cuda:
            return result.cuda()
        else:
            return result



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


