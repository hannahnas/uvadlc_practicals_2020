"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import pickle

import torch
import torch.nn as nn

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100


# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
    
    pred = torch.argmax(predictions, dim=1)
    # tar = torch.argmax(targets, dim=1)
    accuracy = torch.sum(pred == targets) / float(targets.shape[0])
    
    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.
  
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    batch_size = FLAGS.batch_size
    max_steps = FLAGS.max_steps
    eval_freq = FLAGS.eval_freq
    lr = FLAGS.learning_rate

    performance = {'Training loss (batch)': [],
                    'Training accuracy (batch)': [],
                    'Test loss': [],
                    'Test accuracy': []}

    # Load data
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    N_test = cifar10['test'].num_examples

    test_x = torch.Tensor(cifar10['test'].images.reshape(N_test, -1))
    test_y = torch.Tensor(cifar10['test'].labels)
    test_y = torch.argmax(test_y, axis=1)

    # Model and optimization
    n_inputs = 3 * 32 * 32
    n_classes = 10
    mlp = MLP(n_inputs, dnn_hidden_units, n_classes)

    if FLAGS.optim == 'adam':
        optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(mlp.parameters(), lr=lr)
    loss_module = nn.CrossEntropyLoss()

    for i in range(max_steps):
        x, y = cifar10['train'].next_batch(batch_size)
        x, y = torch.Tensor(x).reshape(batch_size, n_inputs), torch.Tensor(y)
        y = torch.argmax(y, axis=1)
        
        pred = mlp(x)
        loss = loss_module(input=pred, target=y)
        performance['Training loss (batch)'].append([i, loss.item()])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation
        if (i % eval_freq == 0) or (i == eval_freq-1):
            print(f'Step: {i}')

            with torch.no_grad():
                test_pred = mlp(test_x)
                test_loss = loss_module(test_pred, test_y)
                test_acc = accuracy(test_pred, test_y)
                train_acc = accuracy(pred, y)
            
            performance['Test loss'].append([i, test_loss])
            performance['Test accuracy'].append([i, test_acc])
            performance['Training accuracy (batch)'].append([i, train_acc])

            print(f'Training loss (batch): {loss}')
            print(f'Test loss: {test_loss}')
            print(f'Train accuracy (batch): {train_acc}')
            print(f'Test accuracy: {test_acc}\n')

    pickle.dump(performance, open("MLP_pytorch_curves_"+FLAGS.optim+".p", "wb" ))


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data'),
    parser.add_argument('--optim', type=str, default='SGD',
                        help='Optimizer')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
