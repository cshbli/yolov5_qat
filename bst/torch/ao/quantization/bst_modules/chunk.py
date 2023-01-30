from __future__ import absolute_import, division, print_function

import torch.nn as nn


class Chunk(nn.Module):
    def __init__(self, chunks, dim=1, validate=True):
        super().__init__()
        self.chunks = chunks
        self.dim = dim
        if validate:
            self.validate()

    def extra_repr(self):
        return 'chunks=%r,dim=%r' % (self.chunks, self.dim)

    def validate(self):
        assert self.dim > 1

    def forward(self, x):
        return x.chunk(self.chunks, dim=self.dim)


class ChunkChannel(Chunk):
    def validate(self):
        assert self.dim == 1


class Split(nn.Module):
    def __init__(self, split, dim=1, validate=True):
        super().__init__()
        self.split = split
        self.dim = dim
        if validate:
            self.validate()

    def extra_repr(self):
        return 'split=%r,dim=%r' % (self.split, self.dim)

    def validate(self):
        assert self.dim > 1

    def forward(self, x):
        return x.split(self.split, dim=self.dim)


class SplitChannel(Split):
    def validate(self):
        assert self.dim == 1
