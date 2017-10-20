
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
from transformer import Transformer

use_cuda = torch.cuda.is_available()


######################################################################
# The Encoder BLSTM Without Embedding (WOE)
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.


class EncoderBLSTM_WOE(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderBLSTM_WOE, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=True)

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
        input = input.unsqueeze(1)
        output, (hn, cn) = self.lstm(input, hidden)
        return output, (hn, cn)

    def initHidden(self):
        result = Variable(torch.zeros(2, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class EncoderBLSTM_WOE_1L(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(EncoderBLSTM_WOE_1L, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=True)
        self.linear = nn.Linear(2*self.hidden_size, self.output_size)

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
        input = input.unsqueeze(1)
        o1, (hn, cn) = self.lstm(input, hidden)
        o1 = o1.squeeze(1)
        output = self.linear(o1)
        return output

    def initHidden(self):
        result = Variable(torch.zeros(2, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
######################################################################
# The Encoder BLSTM With Embedding (WE)
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#


class EncoderBLSTM_WE(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderBLSTM_WE, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=True)

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
        embedded = self.embedding(input).view(1, 1, -1)
        output, (hn, cn) = self.lstm(embedded, hidden)
        return output, (hn, cn)

    def initHidden(self):
        result = Variable(torch.zeros(2, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


######################################################################
# The Encoder Convolutional Bank - BLSTM With Embedding (WE)
# This is Tacotron architectrure without Highway layers (CBL)
#
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.

class EncoderCBL(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, n_layers=1, dropout_p=0.5):
        super(EncoderCBL, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.dropout_p = dropout_p

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

        # BLSTM
        self.lstm = nn.LSTM(self.hidden_size2, self.hidden_size2, bidirectional=True)


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

        # From N X C X L to L X N X C
        ocbn2 = ocbn2.transpose(1,2)
        ocbn2 = ocbn2.transpose(0,1)

        # BLSTM Layer
        output, (hn, cn) = self.lstm(ocbn2, hidden)
        return output, (hn, cn)

    def initHidden(self):
        result = Variable(torch.zeros(2, 1, self.hidden_size2))
        if use_cuda:
            return result.cuda()
        else:
            return result



class EncoderCBHL(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, n_layers=1, dropout_p=0.5):
        super(EncoderCBHL, self).__init__()
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
        
class EncoderCBHL_8L(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, n_layers=1, dropout_p=0.5):
        super(EncoderCBHL_8L, self).__init__()
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


        # Max-Pool Layer
        self.mp = nn.MaxPool1d(2, stride=1, padding=1)

        # Convolution 1-D Projections
        self.cnp1 = nn.Conv1d(self.hidden_size2*8, self.hidden_size2, 3, padding=2)
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


class EncoderCBHL_8L_WOE_1L(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(EncoderCBHL_8L_WOE_1L, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.num_hl_layers = 4
  
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        
        # Convolutional Bank
        self.cn1 = nn.Conv1d(self.hidden_size, self.hidden_size, 1, padding=0)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.cn2 = nn.Conv1d(self.hidden_size, self.hidden_size, 2, padding=1)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)
        self.cn3 = nn.Conv1d(self.hidden_size, self.hidden_size, 3, padding=2)
        self.bn3 = nn.BatchNorm1d(self.hidden_size)
        self.cn4 = nn.Conv1d(self.hidden_size, self.hidden_size, 4, padding=3)
        self.bn4 = nn.BatchNorm1d(self.hidden_size)
        self.cn5 = nn.Conv1d(self.hidden_size, self.hidden_size, 5, padding=4)
        self.bn5 = nn.BatchNorm1d(self.hidden_size)
        self.cn6 = nn.Conv1d(self.hidden_size, self.hidden_size, 6, padding=5)
        self.bn6 = nn.BatchNorm1d(self.hidden_size)
        self.cn7 = nn.Conv1d(self.hidden_size, self.hidden_size, 7, padding=6)
        self.bn7 = nn.BatchNorm1d(self.hidden_size)
        self.cn8 = nn.Conv1d(self.hidden_size, self.hidden_size, 8, padding=7)
        self.bn8 = nn.BatchNorm1d(self.hidden_size)


        # Max-Pool Layer
        self.mp = nn.MaxPool1d(2, stride=1, padding=1)

        # Convolution 1-D Projections
        self.cnp1 = nn.Conv1d(self.hidden_size*8, self.hidden_size, 3, padding=2)
        self.bnp1 = nn.BatchNorm1d(self.hidden_size)
        self.cnp2 = nn.Conv1d(self.hidden_size, self.hidden_size, 3, padding=2)
        self.bnp2 = nn.BatchNorm1d(self.hidden_size)

        # Highway Layers
        self.hw = Highway(self.hidden_size, self.num_hl_layers, f=torch.nn.functional.relu) 

        # BLSTM   
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=True)
        
        # Output Layer
        self.linear2 = nn.Linear(2*self.hidden_size, self.output_size)

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
        po6 = self.linear1(input)
        po6 = self.relu(po6)
        
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
        obl, (hn, cn) = self.lstm(ohl, hidden)
        
        # Final Output Layer
        output = self.linear2(obl.squeeze(1))
        
        return output

    def initHidden(self):
        result = Variable(torch.zeros(2, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
        
        
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
               

    def forward(self, input):
        
        # Embedding Layer
        embedded = self.embedding(input.transpose(0,1)).squeeze(0)
        
        # Transformer Encoder
        output = self.tf(embedded)

        return(output) 
