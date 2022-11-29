import torch
from torch import nn
import torch.nn.functional as F


class FNN(nn.Module):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.activation_list = {"relu": F.relu,
                                "sigmoid": F.sigmoid,
                                "tanh": F.tanh}
        self.activation = self.activation_list[activation]
        self.initializer_list = {"Glorot normal": nn.init.xavier_normal_,
                                 "Glorot uniform": nn.init.xavier_uniform_,
                                 "He normal": nn.init.kaiming_normal_,
                                 "He uniform": nn.init.kaiming_uniform_,
                                 "zeros": nn.init.zeros_}
        self.initializer = self.initializer_list[kernel_initializer]
        self.initializer_zero = self.initializer_list["zeros"]
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i]
                )
            )
            self.initializer(self.linears[-1].weight)
            self.initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        for linear in self.linears[:-1]:
            inputs = self.activation(linear(inputs))
        return self.linears[-1](inputs)
