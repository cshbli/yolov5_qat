from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

import bst.torch.ao.quantization.bst_modules as bstnn

class Sigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = bstnn.Sigmoid()
        self.mul = bstnn.Mul()

    def forward(self, x):
        return self.mul(x, self.sigmoid(x))
