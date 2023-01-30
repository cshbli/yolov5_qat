from __future__ import absolute_import, division, print_function

import torch.nn as nn


class Repeat(nn.Module):
    def __init__(self, sizes, validate=True):
        super().__init__()
        self.sizes = sizes
        if validate:
            self.validate()

    def extra_repr(self):
        return 'sizes=%r' % (self.sizes,)

    def validate(self):
        assert self.sizes[1] == 1

    def forward(self, x):
        return x.repeat(*self.sizes)


class RepeatChannel(Repeat):
    def validate(self):
        assert self.sizes[1] != 1


class RepeatInterleave(nn.Module):
    def __init__(self, repeats, dim=None, validate=True):
        super().__init__()
        self.repeats = repeats
        self.dim = dim
        if validate:
            self.validate()

    def extra_repr(self):
        return 'repeats=%s,dim=%s' % (self.repeats, self.dim)

    def validate(self):
        assert self.repeats == 1 or self.dim != 1

    def forward(self, x):
        return x.repeat_interleave(self.repeats, dim=self.dim)


class RepeatInterleaveChannel(RepeatInterleave):
    def validate(self):
        assert self.repeats != 1 and self.dim == 1
