from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class Cat(nn.Module):
    def __init__(self, dim=1, validate=True):
        super().__init__()
        self.dim = dim
        if validate:
            self.validate()

    def extra_repr(self):
        return 'dim=%s' % self.dim

    def validate(self):
        assert self.dim > 1

    def forward(self, *tensors):
        return torch.cat(tensors, self.dim)


class CatChannel(nn.Module):    
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *tensors):
        return torch.cat(tensors, self.dim)
