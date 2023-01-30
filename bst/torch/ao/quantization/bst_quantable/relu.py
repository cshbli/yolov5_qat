from __future__ import absolute_import, division, print_function

import torch.nn as nn
from torch import Tensor

class ReLU(nn.ReLU):
    _FLOAT_MODULE = nn.ReLU

    def __init__(self, qconfig=None, **kwargs) -> None:
        super(ReLU, self).__init__(**kwargs)
        assert qconfig, 'qconfig must be provided for quantable module'
        self.qconfig = qconfig

    def forward(self, inputs: Tensor) -> Tensor:
        return super(ReLU, self).forward(inputs)
    
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
        quantable_relu = cls(qconfig=qconfig)
        return quantable_relu