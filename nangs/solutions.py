# AUTOGENERATED! DO NOT EDIT! File to edit: 02_solutions.ipynb (unless otherwise specified).

__all__ = ['get_activation', 'block', 'MLP']

# Cell

import torch

def get_activation(a):
    if a == "relu": return torch.nn.ReLU(inplace=True)
    elif a == "sigmoid": return torch.nn.Sigmoid()
    else: raise Exception("invalid activation")

def block(i, o, a):
    return torch.nn.Sequential(
        torch.nn.Linear(i, o),
        get_activation(a)
    )

class MLP(torch.nn.Module):
    def __init__(self, inputs, outputs, layers, neurons, activations="relu"):
        super().__init__()
        self.fc_in = block(inputs, neurons, activations)
        self.fc_hidden = torch.nn.ModuleList()
        for layer in range(layers):
            self.fc_hidden.append(block(neurons, neurons, activations))
        self.fc_out = torch.nn.Linear(neurons, outputs, activations)

    def forward(self, x):
        x = self.fc_in(x)
        for layer in self.fc_hidden:
            x = layer(x)
        x = self.fc_out(x)
        return x