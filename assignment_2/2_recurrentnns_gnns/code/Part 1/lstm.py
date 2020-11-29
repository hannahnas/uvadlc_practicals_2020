"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn.functional as F
import torch.nn as nn
import torch


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTM, self).__init__()
        
        self.embed_dim = int(hidden_dim/2)
        self.embedding = nn.Embedding(num_classes, self.embed_dim).to(device)
        self.hidden = hidden_dim
        self.device = device

        self.W_gx = self.weights(hidden_dim, self.embed_dim, device, 'linear')
        self.W_ix = self.weights(hidden_dim, self.embed_dim, device, 'linear') 
        self.W_fx = self.weights(hidden_dim, self.embed_dim, device, 'linear')
        self.W_ox = self.weights(hidden_dim, self.embed_dim, device, 'linear')

        self.W_gh = self.weights(hidden_dim, hidden_dim, device, 'tanh')
        self.W_ih = self.weights(hidden_dim, hidden_dim, device, 'tanh') 
        self.W_fh = self.weights(hidden_dim, hidden_dim, device, 'tanh')
        self.W_oh = self.weights(hidden_dim, hidden_dim, device, 'tanh') 

        self.b_g = self.bias(hidden_dim, device)
        self.b_i = self.bias(hidden_dim, device)
        self.b_f = self.bias(hidden_dim, device) 
        self.b_o = self.bias(hidden_dim, device)

        self.W_ph = self.weights(num_classes, hidden_dim, device, 'tanh')
        self.b_p = self.bias(num_classes, device)


    def weights(self, dim_1, dim_2, device, nonlin='linear'):
        W = torch.Tensor(dim_1, dim_2).to(device)
        nn.init.kaiming_normal_(W, nonlinearity=nonlin)
        W = nn.Parameter(W)
        return W

    def bias(self, dim, device):
        b = nn.Parameter(torch.zeros(dim, 1)).to(device)
        return b

    def forward(self, x):
        h = torch.zeros((self.hidden, 1), requires_grad=True).to(self.device) 
        c = torch.zeros((self.hidden, 1), requires_grad=True).to(self.device)

        batch, time = x.shape
        x = torch.unsqueeze(x, 2)

        for t in range(time):

            x_t = self.embedding(x[:, t, :].long())
            x_t = torch.squeeze(x_t).T

            g = torch.tanh(self.W_gx @ x_t + self.W_gh @ h + self.b_g)
            i = torch.sigmoid(self.W_ix @ x_t + self.W_ih @ h + self.b_i)
            f = torch.sigmoid(self.W_fx @ x_t + self.W_fh @ h + self.b_f)
            o = torch.sigmoid(self.W_ox @ x_t + self.W_oh @ h + self.b_o)
            c = g * i + c * f
            h = torch.tanh(c) * o
        
        p = self.W_ph @ h + self.b_p
        y = F.log_softmax(p, dim=0)
        y = y.T

        return y
