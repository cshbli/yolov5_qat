from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class Mean(nn.Module):
    def __init__(self, dim, keepdim=False, validate=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        if validate:
            self.validate()

    def extra_repr(self):
        return 'dim=%r,keepdim=%r' % (self.dim, self.keepdim)

    def validate(self):
        assert self.dim != 1

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


class MeanChannel(Mean):
    def validate(self):
        assert self.dim == 1


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class Min(nn.Module):
    def __init__(self, dim, keepdim=False, validate=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        if validate:
            self.validate()

    def extra_repr(self):
        return 'dim=%r,keepdim=%r' % (self.dim, self.keepdim)

    def validate(self):
        assert self.dim != 1

    def forward(self, x):
        return torch.amin(x, dim=self.dim, keepdim=self.keepdim)


class MinChannel(Min):
    def validate(self):
        assert self.dim == 1


class Max(nn.Module):
    def __init__(self, dim, keepdim=False, validate=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        if validate:
            self.validate()

    def extra_repr(self):
        return 'dim=%r,keepdim=%r' % (self.dim, self.keepdim)

    def validate(self):
        assert self.dim != 1

    def forward(self, x):
        return torch.max(x, dim=self.dim, keepdim=self.keepdim)[0]


class MaxChannel(Max):
    def validate(self):
        assert self.dim == 1


class Median(nn.Module):
    def __init__(self, dim, keepdim=False, validate=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        if validate:
            self.validate()

    def extra_repr(self):
        return 'dim=%r,keepdim=%r' % (self.dim, self.keepdim)

    def validate(self):
        assert self.dim != 1

    def forward(self, x):
        return torch.median(x, dim=self.dim, keepdim=self.keepdim)


class MedianChannel(Median):
    def validate(self):
        assert self.dim == 1


class Sum(nn.Module):
    def __init__(self, dim, keepdim=False, validate=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        if validate:
            self.validate()

    def extra_repr(self):
        return 'dim=%r,keepdim=%r' % (self.dim, self.keepdim)

    def validate(self):
        assert self.dim != 1

    def forward(self, x):
        return torch.sum(x, dim=self.dim, keepdim=self.keepdim)


class SumChannel(Sum):
    def validate(self):
        assert self.dim == 1
