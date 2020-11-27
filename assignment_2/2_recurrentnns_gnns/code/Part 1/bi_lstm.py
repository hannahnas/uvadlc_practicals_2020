"""
This module implements a bidirectional LSTM in PyTorch.
You should fill in code into indicated sections.
Date: 2020-11-09
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn.functional as F
import torch.nn as nn
import torch

def weights(dim_1, dim_2, device, nonlin='linear'):
    W = nn.Parameter(torch.Tensor(dim_1, dim_2))
    nn.init.kaiming_normal_(W, nonlinearity=nonlin)
    return W.to(device)

def bias(dim, device):
    b = nn.Parameter(torch.zeros(dim, 1))
    return b.to(device)


class biLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(biLSTM, self).__init__()
        
        self.device = device
        embed_dim = int(hidden_dim/2)
        self.embedding = nn.Embedding(num_classes, embed_dim).to(device)    

        self.forward_layer = LSTMCell(seq_length, embed_dim, hidden_dim, num_classes, batch_size, device)
        self.backward_layer = LSTMCell(seq_length, embed_dim, hidden_dim, num_classes, batch_size, device)

        self.W_ph = weights(num_classes, 2*hidden_dim, device, 'tanh')
        self.b_p = bias(num_classes, device)

        self.h = torch.zeros((hidden_dim, 1), requires_grad=True).to(device) 
        self.c = torch.zeros((hidden_dim, 1), requires_grad=True).to(device)


    def forward(self, x):
        x_reverse = torch.flip(x, [1])
        x = self.embedding(x.long())
        x_reverse = self.embedding(x_reverse.long())

        # forward layer 
        h = self.h
        c = self.c
        h_t = self.forward_layer(x, c, h)

        # backward layer
        h = self.h
        c = self.c
        h_0 = self.backward_layer(x_reverse, c, h)

        H = torch.cat((h_t, h_0))

        p = self.W_ph @ H + self.b_p
        y = F.log_softmax(p, dim=0)
        y = y.T
        return y


class LSTMCell(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTMCell, self).__init__()

        self.W_gx = weights(hidden_dim, input_dim, device, 'linear')
        self.W_ix = weights(hidden_dim, input_dim, device, 'linear')
        self.W_fx = weights(hidden_dim, input_dim, device, 'linear')
        self.W_ox = weights(hidden_dim, input_dim, device, 'linear')

        self.W_gh = weights(hidden_dim, hidden_dim, device, 'tanh')
        self.W_ih = weights(hidden_dim, hidden_dim, device, 'tanh')
        self.W_fh = weights(hidden_dim, hidden_dim, device, 'tanh')
        self.W_oh = weights(hidden_dim, hidden_dim, device, 'tanh')

        self.b_g = bias(hidden_dim, device)
        self.b_i = bias(hidden_dim, device)
        self.b_f = bias(hidden_dim, device)
        self.b_o = bias(hidden_dim, device)
 

    def forward(self, x, c, h):
        batch, time, embed = x.shape

        for t in range(time):
            x_t = torch.squeeze(x[:, t, :]).T

            g = torch.tanh(self.W_gx @ x_t + self.W_gh @ h + self.b_g)
            i = torch.sigmoid(self.W_ix @ x_t + self.W_ih @ h + self.b_i)
            f = torch.sigmoid(self.W_fx @ x_t + self.W_fh @ h + self.b_f)
            o = torch.sigmoid(self.W_ox @ x_t + self.W_oh @ h + self.b_o)
            c = g * i + c * f
            h = torch.tanh(c) * o

        return h
