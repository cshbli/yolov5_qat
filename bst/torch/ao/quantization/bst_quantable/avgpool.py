from __future__ import absolute_import, division, print_function

import numpy as np

import torch.nn as nn
from torch import Tensor

from bst.torch.ao.quantization import bst_modules as bstnn

class A1000A0AvgPool2d(bstnn.A1000A0AvgPool2d):    
    r"""
    A A1000A0AvgPool2d module used for quantization aware training.
    We adopt the same interface as `torch.nn.AvgPool2d`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d
    for documentation.
    Similar to `torch.nn.AvgPool2d`, with FakeQuantize modules initialized to
    default.
    """
    _FLOAT_MODULE = bstnn.A1000A0AvgPool2d

    def __init__(self, qconfig=None,
                 device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(A1000A0AvgPool2d, self).__init__(**kwargs)
        assert qconfig, 'qconfig must be provided for quantable module'
        self.qconfig = qconfig

    def forward(self, input: Tensor) -> Tensor:
        return super(A1000A0AvgPool2d, self).forward(input)
    
    @classmethod
    def from_float(cls, mod, qconfig=None):
        r"""Create a quantable module from a float module or qparams_dict
            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        quantable_avgpool2d = cls(kernel_size=mod.kernel_size,
                       stride=mod.stride, padding=mod.padding, qconfig=qconfig)
        return quantable_avgpool2d