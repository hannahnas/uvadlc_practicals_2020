"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch import nn


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        
        super(ConvNet, self).__init__()

        self.conv_0 = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)   

        self.pre_act_1 = self.pre_act(channels=64)

        self.part_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )

        self.pre_act_2a = self.pre_act(channels=128)
        self.pre_act_2b = self.pre_act(channels=128)

        self.part_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )

        self.pre_act_3a = self.pre_act(channels=256)
        self.pre_act_3b = self.pre_act(channels=256)

        self.part_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )

        self.pre_act_4a = self.pre_act(channels=512)
        self.pre_act_4b = self.pre_act(channels=512)

        self.maxpool_4 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.pre_act_5a = self.pre_act(channels=512)
        self.pre_act_5b = self.pre_act(channels=512)

        self.max_pool_5 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.linear = nn.Sequential(
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_classes)
        )


    def pre_act(self, channels):

        pre_act = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=1, stride=1)
        )

        return pre_act  
    
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
        
        out = self.conv_0(x)

        out = out + self.pre_act_1(out)

        out = self.part_1(out)

        out = out + self.pre_act_2a(out)
        out = out + self.pre_act_2a(out)

        out = self.part_2(out)

        out = out + self.pre_act_3a(out)
        out = out + self.pre_act_3b(out)

        out = self.part_3(out)

        out = out + self.pre_act_4a(out)
        out = out + self.pre_act_4b(out)

        out = self.maxpool_4(out)

        out = out + self.pre_act_5a(out)
        out = out + self.pre_act_5a(out)

        out = self.max_pool_5(out)
        out = self.linear(out.squeeze())

        return out
