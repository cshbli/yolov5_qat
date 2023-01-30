from __future__ import absolute_import, division, print_function

import torch.nn as nn


class Flatten(nn.Flatten):
    def __init__(self, start_dim=1, end_dim=-1, validate=True):
        super().__init__(start_dim=start_dim, end_dim=end_dim)
        if validate:
            self.validate()

    def validate(self):
        if self.start_dim >= 0:
            assert self.start_dim > 1


class FlattenChannel(Flatten):
    def validate(self):
        assert self.start_dim >= 0 and self.start_dim <= 1


class Unflatten(nn.Unflatten):
    def __init__(self, dim, unflattened_size, validate=True):
        super().__init__(dim=dim, unflattened_size=unflattened_size)
        if validate:
            self.validate()

    def validate(self):
        assert self.dim != 1


class UnflattenChannel(Unflatten):
    def validate(self):
        assert self.dim == 1
