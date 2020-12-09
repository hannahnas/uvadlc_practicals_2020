# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        
        self.seq_length = seq_length
        self.model = nn.LSTM(vocabulary_size, lstm_num_hidden, lstm_num_layers, batch_first=False).to(device)
        self.W_ph = nn.Parameter(torch.Tensor(lstm_num_hidden, vocabulary_size)).to(device)
        self.b_p = nn.Parameter(torch.zeros(1, 1, vocabulary_size)).to(device)
        nn.init.kaiming_normal_(self.W_ph, nonlinearity='tanh')

    def forward(self, x):
        out, _ = self.model(x)
        out = torch.einsum("abc,cd->abd", (out, self.W_ph)) + self.b_p
        # out = F.softmax(out, dim=2)

        return out

