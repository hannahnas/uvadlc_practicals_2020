"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import pickle

import torch
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
    accuracy = torch.sum(pred == targets) / float(targets.shape[0])
 
    return accuracy


def train():
    """
    Performs training and evaluation of ConvNet model.
  
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    max_steps = FLAGS.max_steps
    lr = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    eval_freq = FLAGS.eval_freq

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", device)

     # Load data
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    N_test = cifar10['test'].num_examples
    test_batch_size = 50

    # Initialize network
    cnn = ConvNet(n_channels=3, n_classes=10).to(device)

    # print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    loss_module = nn.CrossEntropyLoss()
    
    performance = {}
    performance['Training loss (batch)'] = []
    performance['Training accuracy (batch)'] = []
    performance['Test loss'] = []
    performance['Test accuracy'] = []

    for i in range(max_steps):
        x, y = cifar10['train'].next_batch(batch_size)
        x, y = torch.Tensor(x).to(device), torch.Tensor(y).to(device)
        y = torch.argmax(y, axis=1)
        
        pred = cnn(x)
        loss = loss_module(input=pred, target=y)
        performance['Training loss (batch)'].append([i, loss.item()])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % eval_freq == 0 or i == 4999:
            print(f'Step: {i}')

            # calculate accuracy
            test_acc = 0
            test_loss = 0
            
            runs = N_test/test_batch_size

            with torch.no_grad():
                for _ in range(int(runs)):

                    x_test, y_test = cifar10['test'].next_batch(test_batch_size)
                    x_test, y_test = torch.Tensor(x_test).to(device), torch.Tensor(y_test).to(device)
                    y_test = torch.argmax(y_test, axis=1)

                    test_pred = cnn(x_test)
                    test_acc += accuracy(test_pred, y_test)
                    test_loss += loss_module(input=test_pred, target=y_test).item()

            test_acc = test_acc / (runs)
            test_loss = test_loss / (runs)
            train_acc = accuracy(pred, y)

            performance['Test loss'].append([i, test_loss])
            performance['Test accuracy'].append([i, test_acc])
            performance['Training accuracy (batch)'].append([i, train_acc])

            print(f'Training loss (batch): {loss}')
            print(f'Training accuracy (batch): {train_acc}')
            print(f'Test loss: {test_loss}')
            print(f'Test accuracy: {test_acc}\n')

    pickle.dump(performance, open("CNN_curves.p", "wb" ))



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
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
