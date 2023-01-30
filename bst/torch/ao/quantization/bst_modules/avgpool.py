from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

class A1000A0AvgPool2d(nn.AvgPool2d):
    _ORIGIN_MODULE = nn.AvgPool2d  # for inplace replacement of with unmodified module

    def __init__(self, **kwargs) -> None:
        super(A1000A0AvgPool2d, self).__init__(**kwargs)

        kernel_shape = kwargs.get('kernel_size', None)
        assert kernel_shape is not None, "Avgpool kernel size cannot be None"

        divisor_size = kernel_shape[0] * kernel_shape[1]
        
        self.kwargs = kwargs

        # For A1000A0 Int8 only
        self.left_shift = np.floor(1 / divisor_size * 2**8) * divisor_size
        self.right_shift = 1 / 2**8

    def forward(self, input: Tensor) -> Tensor:
        output = F.avg_pool2d(input, self.kernel_size, self.stride,
                                self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)

        if self.left_shift * self.right_shift != 1.0:
            output = output * self.left_shift * self.right_shift
        
        return output

    def to_origin(self, **kwargs) -> nn.Module:
        # step1: create origin module
        origin_module = self._ORIGIN_MODULE(**kwargs)

        # step2: copy over any parameters. Eg. weights, bias
        
        # step3: return origin module
        return origin_module