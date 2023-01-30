from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class Mul(nn.Module):
    def forward(self, x, y):
        return torch.mul(x, y)
