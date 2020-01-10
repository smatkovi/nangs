# AUTOGENERATED! DO NOT EDIT! File to edit: 02_solution.ipynb (unless otherwise specified).

__all__ = ['Solution', 'getActivation', 'block']

# Cell

import torch
import torch.nn as nn

class Solution(nn.Module):
    "Currently, only MLPs are supported as solution approximators wiht same number of neurons and activation \
    function per layer"
    def __init__(self, inputs, outputs, layers, neurons, activations):
        super().__init__()

        # checks
        if not isinstance(inputs, int) or inputs <= 0: raise Exception('inputs must be a postive integer')
        if not isinstance(outputs, int) or outputs <= 0: raise Exception('outputs must be a positive integer')
        if not isinstance(layers, int) or layers <= 0: raise Exception('layers must be a positive integer')
        if not isinstance(neurons, int) or neurons <= 0: raise Exception('neurons must be a positive integer')
        if not isinstance(activations, str): raise Exception('activation must be a string')

        # activaton function
        self.activation = activations
        # layers
        self.fc_in = block(inputs, neurons, self.activation)
        self.fc_hidden = nn.ModuleList()
        for layer in range(layers):
            self.fc_hidden.append(block(neurons, neurons, self.activation))
        self.fc_out = nn.Linear(neurons, outputs)


    def forward(self, x):
        x = self.fc_in(x)
        for layer in self.fc_hidden:
            x = layer(x)
        x = self.fc_out(x)
        return x


def getActivation(a):
    if a == 'relu': return nn.ReLU(inplace=True)
    elif a == 'sigmoid': return nn.Sigmoid(inplace=True)
    else: raise Exception(f'activation function {a} not valid')

def block(i, o, a):
    return nn.Sequential(
        nn.Linear(i, o),
        #nn.BatchNorm1d(o),
        getActivation(a)
    )