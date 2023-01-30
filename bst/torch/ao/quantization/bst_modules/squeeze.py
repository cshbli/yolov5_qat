from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class Squeeze(nn.Module):
    def __init__(self, dim=None, validate=True):
        super().__init__()
        self.dim = dim
        if validate:
            self.validate()

    def extra_repr(self):
        return 'dim=%r' % self.dim

    def validate(self):
        assert self.dim is not None
        assert self.dim != 1

    def forward(self, x):
        if self.dim is None:
            return torch.squeeze(x)
        else:
            return torch.squeeze(x, dim=self.dim)


class SqueezeChannel(Squeeze):
    def validate(self):
        assert self.dim is None or self.dim == 1


class Unsqueeze(nn.Module):
    def __init__(self, dim, validate=True):
        super().__init__()
        self.dim = dim
        if validate:
            self.validate()

    def extra_repr(self):
        return 'dim=%r' % self.dim

    def validate(self):
        assert self.dim != 1

    def forward(self, x):
        return torch.unsqueeze(x, dim=self.dim)


class UnsqueezeChannel(Unsqueeze):
    def validate(self):
        assert self.dim == 1
