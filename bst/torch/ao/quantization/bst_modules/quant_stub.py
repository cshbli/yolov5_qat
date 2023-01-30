from __future__ import absolute_import, division, print_function

import torch.nn as nn


class QuantStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_value', None)

    def set_value(self, value):
        self._value = value

    def forward(self):
        return self._value
