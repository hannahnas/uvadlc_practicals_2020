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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

###############################################################################


def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size)
    vocab = dataset.vocab_size

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, vocab)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # FIXME
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)#, weight_decay=config.learning_rate_decay)

    batch = config.batch_size

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        batch_inputs = torch.stack(batch_inputs)
        batch_targets = torch.stack(batch_targets).to(device)

        batch_inputs = F.one_hot(batch_inputs, num_classes=vocab).float().to(device)
        targets_onehot = F.one_hot(batch_targets, num_classes=vocab).to(device)

        model.zero_grad()
        pred = model(batch_inputs)     
        
        loss = cross_entropy_loss(pred, targets_onehot)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.max_norm)
        optimizer.step()

        accuracy = batch_accuracy(pred, batch_targets)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if (step + 1) % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                    Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))

        if (step + 1) % config.sample_every == 0:
            # Generate 5 sentences by sampling from the model
            letters = torch.randint(high=vocab, size=(1, 5)).to(device)
            sentence = [letters]
            letters = F.one_hot(letters, num_classes=vocab).float().to(device)
            with torch.no_grad():
                for _ in range(config.seq_length-1):
                    probs = model(letters)[-1, :, :]
                    probs = torch.unsqueeze(probs, dim=0)
                    next_letters = torch.argmax(probs, dim=2)
                    sentence.append(next_letters)
                    next_letters = F.one_hot(next_letters, num_classes=vocab)
                    letters = torch.cat((letters, next_letters), dim=0)

            sentences = torch.stack(sentence)
            print(sentences.shape)
            for i in range(5):
                sentence = sentences[:, 0, i].tolist()
                string = dataset.convert_to_string(sentence)
                print(f"sentence {i}: {string}")

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error,
            # check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

def cross_entropy_loss(pred, target):
    # predictions of size (sequence_length, batch_size, vocabulary_size)
    # targets of size (sequence_length, batch_size, vocabulary_size)
    L = - torch.sum(target * torch.log(pred), dim=2)
    L = torch.mean(L, dim=0)
    return torch.mean(L)

def batch_accuracy(pred, target):
    # predictions of size (sequence_length, batch_size, vocabulary_size)
    # targets of size (sequence_length, batch_size)
    pred = torch.argmax(pred, dim=2)
    accuracy = torch.mean((pred == target).float(), dim=(0,1))
    return accuracy
    


###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True,
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6),
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # Train the model
    train(config)
