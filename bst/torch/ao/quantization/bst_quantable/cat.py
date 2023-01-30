from __future__ import absolute_import, division, print_function

import numpy as np

import torch.nn as nn
from torch import Tensor

from bst.torch.ao.quantization import bst_modules as bstnn

class Cat(bstnn.Cat):
    _FLOAT_MODULE = bstnn.Cat

    def __init__(self, qconfig=None, **kwargs) -> None:
        super(Cat, self).__init__(**kwargs)
        assert qconfig, 'qconfig must be provided for quantable module'
        self.qconfig = qconfig
        self.activation_post_process = qconfig.activation()

    def forward(self, *tensors) -> Tensor:
        return self.activation_post_process(super(Cat, self).forward(*tensors))
    
    @classmethod
    def from_float(cls, mod):
        r"""Create a quantable module from a float module or qparams_dict
            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        quantable_cat = cls(qconfig=qconfig)
        return quantable_cat

class CatChannel(bstnn.CatChannel):
    _FLOAT_MODULE = bstnn.CatChannel

    def __init__(self, qconfig=None, **kwargs) -> None:
        super(CatChannel, self).__init__(**kwargs)
        assert qconfig, 'qconfig must be provided for quantable module'
        self.qconfig = qconfig

    def forward(self, *tensors) -> Tensor:
        return super(CatChannel, self).forward(*tensors)
    
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
        quantable_cat = cls(qconfig=qconfig)
        return quantable_cat