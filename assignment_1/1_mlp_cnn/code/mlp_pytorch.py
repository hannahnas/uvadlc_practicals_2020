"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.
        
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
    
        TODO:
        Implement initialization of the network.
        """
        super(MLP, self).__init__()

        nodes = [n_inputs] + n_hidden

        iterable = list(zip(nodes, nodes[1:]))

        layers = []
        for inputs, outputs in iterable:
            linear = nn.Linear(inputs, outputs)
            
            with torch.no_grad():
                weights = torch.normal(mean=0.0, std=0.0001, size=(outputs, inputs))
                linear.weight.copy_(weights)
                linear.bias.zero_()
            
            layers.append(linear)
            layers.append(nn.ELU())

        layers.append(nn.Linear(nodes[-1], n_classes))

        self.network = nn.Sequential(*layers)

    
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        TODO:
        Implement forward pass of the network.
        """
        
        out = self.network(x)
        
        return out
