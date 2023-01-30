

import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from bst.torch.ao.quantization import bst_modules as bstnn

class ConvTranspose2d(nn.ConvTranspose2d):
    r"""
    A ConvTranspose2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.ConvTranspose2d`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
    for documentation.

    Similar to `torch.nn.ConvTranspose2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nn.ConvTranspose2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, output_padding=0, groups=1, 
                 bias=True, padding_mode='zeros', qconfig=None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation, output_padding=output_padding,
                         groups=groups, bias=bias, padding_mode=padding_mode,
                         **factory_kwargs)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore[arg-type]
        return F.conv_transpose2d(
            input, self.weight_fake_quant(self.weight), self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        
    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        qat_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                       stride=mod.stride, padding=mod.padding, dilation=mod.dilation, output_padding=mod.output_padding,
                       groups=mod.groups, bias=mod.bias is not None,
                       padding_mode=mod.padding_mode, qconfig=qconfig)
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv
