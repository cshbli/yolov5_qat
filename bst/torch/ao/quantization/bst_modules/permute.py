from __future__ import absolute_import, division, print_function

import torch.nn as nn


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def extra_repr(self):
        return 'dims=%r' % (self.dims,)

    def forward(self, x):
        return x.permute(*self.dims)
