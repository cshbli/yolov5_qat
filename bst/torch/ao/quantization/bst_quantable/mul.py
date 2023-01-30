from __future__ import absolute_import, division, print_function

import numpy as np

import torch.nn as nn
from torch import Tensor

from bst.torch.ao.quantization import bst_modules as bstnn

class Mul(bstnn.Mul):
    _FLOAT_MODULE = bstnn.Mul

    def __init__(self, qconfig=None, **kwargs) -> None:
        super(Mul, self).__init__(**kwargs)
        assert qconfig, 'qconfig must be provided for quantable module'
        self.qconfig = qconfig

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        return super(Mul, self).forward(input1, input2)
    
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
        quantable_mul = cls(qconfig=qconfig)
        return quantable_mul