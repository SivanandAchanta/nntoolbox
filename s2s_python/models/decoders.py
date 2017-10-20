#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:06:36 2017

@author: Sivanand Achanta
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

class AttnDecoderGRU1L_SMAXOP(nn.Module):
    def __init__(self, hidden_size1, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(AttnDecoderGRU1L_SMAXOP, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size1 = hidden_size1
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.prenet = nn.Linear(self.output_size, self.hidden_size1)
        self.prenet2 = nn.Linear(self.hidden_size1, self.hidden_size)
        self.attn1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attn2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(self.hidden_size, 1, bias=False)
        self.smax1 = nn.Softmax()
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size*2, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.out = nn.Softmax()

    def forward(self, input, hidden, encoder_outputs):

        em1 = self.prenet(input)
        em2 = self.relu(em1)
        em3 = self.dropout(em2)

        em4 = self.prenet2(em3)
        em5 = self.relu(em4)
        em6 = self.dropout(em5)

        way = self.attn1(em6)
        uah = self.attn2(encoder_outputs)
        #print(way)
        #print(uah)
        #print(way + uah)

        beta = self.tanh(way.expand_as(uah) + uah)
        gamma = self.attn3(beta)
        #print(gamma)
        attn_weights = self.smax1(gamma.transpose(0,1))
        #print(attn_weights)
        #print(attn_weights.size())
        #print(attn_weights.sum(1))
        attn_applied = torch.mm(attn_weights,
                                 encoder_outputs)

        #attn_applied = attn_applied.squeeze(0)

        o1 = torch.cat((em6, attn_applied), 1)
        #print(o1.size())
        o2 = o1.unsqueeze(0)

        #print(o2.size())

        #for i in range(self.n_layers):
        output, hidden = self.gru(o2, hidden)

        #print(output[0])
        o3 = self.linear(output[0])
        output = self.out(o3)

        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderLSTM3L_R1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(AttnDecoderLSTM3L_R1, self).__init__()
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
        self.attn2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(self.hidden_size, 1, bias=False)
        self.smax1 = nn.Softmax()
        self.lstm1 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.lstm2 = nn.LSTM(self.hidden_size*2, self.hidden_size)
        self.lstm3 = nn.LSTM(self.hidden_size*2, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
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
        o1, (h1, c1) = self.lstm1(po6.unsqueeze(0), (h1, c1))
        o2 = o1.squeeze(0)

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
        o4 = o3.unsqueeze(0)
        o5, (h2, c2) = self.lstm2(o4, (h2, c2))

        # LSTM - 3
        o6 = torch.cat((o5.squeeze(0), attn_applied), 1)
        o7 = o6.unsqueeze(0)
        o8, (h3, c3) = self.lstm3(o7, (h3, c3))

        # Output Layer
        output = self.linear(o8[0])

        return output, h1, c1, h2, c2, h3, c3, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderLSTM3L_R2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(AttnDecoderLSTM3L_R2, self).__init__()
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
        o6 = torch.cat((o5.squeeze(1), attn_applied), 1)
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

class AttnDecoderLSTM3L_R2_Rescon(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(AttnDecoderLSTM3L_R2_Rescon, self).__init__()
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

        o6 = o61 + o3
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

############################### R = 3 ############################################
class AttnDecoderLSTM3L_R3_Rescon(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(AttnDecoderLSTM3L_R3_Rescon, self).__init__()
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
        self.linear3 = nn.Linear(self.hidden_size, self.output_size)


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

        o6 = o61 + o3
        o7 = o6.unsqueeze(1)
        o8, (h3, c3) = self.lstm3(o7, (h3, c3))
        o9 = o8.squeeze(1)

        # Output Layer
        o10 = self.linear1(o9)
        o11 = self.linear2(o9)
        o12 = self.linear3(o9)
        output = torch.cat((o10, o11, o12), 1)

        return output, o12, h1, c1, h2, c2, h3, c3, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


############################## R = 4 ####################################################
class AttnDecoderLSTM3L_R4_Rescon(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(AttnDecoderLSTM3L_R4_Rescon, self).__init__()
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
        self.linear3 = nn.Linear(self.hidden_size, self.output_size)
        self.linear4 = nn.Linear(self.hidden_size, self.output_size)


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

        o6 = o61 + o3
        o7 = o6.unsqueeze(1)
        o8, (h3, c3) = self.lstm3(o7, (h3, c3))
        o9 = o8.squeeze(1)

        # Output Layer
        o10 = self.linear1(o9)
        o11 = self.linear2(o9)
        o12 = self.linear3(o9)
        o13 = self.linear4(o9)
        output = torch.cat((o10, o11, o12, o13), 1)

        return output, o13, h1, c1, h2, c2, h3, c3, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


############################## R = 4 With Windowed Attention ####################################################
class AttnDecoderLSTM3L_R4_Rescon_WinATT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(AttnDecoderLSTM3L_R4_Rescon_WinATT, self).__init__()
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
        self.linear3 = nn.Linear(self.hidden_size, self.output_size)
        self.linear4 = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, h1, c1, h2, c2, h3, c3, encoder_outputs, win_len, dec_ix, sl_dec):


            
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

        o6 = o61 + o3
        o7 = o6.unsqueeze(1)
        o8, (h3, c3) = self.lstm3(o7, (h3, c3))
        o9 = o8.squeeze(1)

        # Output Layer
        o10 = self.linear1(o9)
        o11 = self.linear2(o9)
        o12 = self.linear3(o9)
        o13 = self.linear4(o9)
        output = torch.cat((o10, o11, o12, o13), 1)

        return output, o13, h1, c1, h2, c2, h3, c3, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
####################################### R = 5 #########################################
class AttnDecoderLSTM3L_R5_Rescon(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.5):
        super(AttnDecoderLSTM3L_R5_Rescon, self).__init__()
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
        self.linear3 = nn.Linear(self.hidden_size, self.output_size)
        self.linear4 = nn.Linear(self.hidden_size, self.output_size)
        self.linear5 = nn.Linear(self.hidden_size, self.output_size)

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

        o6 = o61 + o3
        o7 = o6.unsqueeze(1)
        o8, (h3, c3) = self.lstm3(o7, (h3, c3))
        o9 = o8.squeeze(1)

        # Output Layer
        o10 = self.linear1(o9)
        o11 = self.linear2(o9)
        o12 = self.linear3(o9)
        o13 = self.linear4(o9)
        o14 = self.linear5(o9)
        output = torch.cat((o10, o11, o12, o13, o14), 1)

        return output, o14, h1, c1, h2, c2, h3, c3, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

