from __future__ import absolute_import, division, print_function

import torch
from torch import nn
from torch.nn import functional as F


class IdConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def extra_repr(self):
        return 'channels=%r' % 10

    def _forward(self, input, weight):
        raise NotImplementedError

    def forward(self, input):
        return self._forward(input, self.weight)


class IdConv1d(IdConv):
    def __init__(self, channels):
        super().__init__(channels)
        weight = torch.eye(channels, dtype=torch.float)
        weight = weight.reshape(channels, channels, 1)
        self.register_buffer('weight', weight)

    def _forward(self, input, weight):
        return F.conv1d(input, weight)


class IdConv2d(IdConv):
    def __init__(self, channels):
        super().__init__(channels)
        weight = torch.eye(channels, dtype=torch.float)
        weight = weight.reshape(channels, channels, 1, 1)
        self.register_buffer('weight', weight)

    def _forward(self, input, weight):
        return F.conv2d(input, weight)


class IdConv3d(IdConv):
    def __init__(self, channels):
        super().__init__(channels)
        weight = torch.eye(channels, dtype=torch.float)
        weight = weight.reshape(channels, channels, 1, 1, 1)
        self.register_buffer('weight', weight)

    def _forward(self, input, weight):
        return F.conv3d(input, weight)
