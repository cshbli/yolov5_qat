from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class Add(nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)
