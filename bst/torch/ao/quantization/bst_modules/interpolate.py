from __future__ import absolute_import, division, print_function

import torch.nn as nn
from torch.nn.functional import interpolate


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def extra_repr(self):
        return 'size=%r,scale_factor=%r,mode=%s' % (self.size, self.scale_factor, self.mode)

    def forward(self, x):
        return interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners, self.recompute_scale_factor)


class InterpolateNearest(Interpolate):
    def __init__(self, size=None, scale_factor=None, align_corners=None, recompute_scale_factor=None):
        super().__init__(size=size, scale_factor=scale_factor, mode='nearest',
                         align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
