from __future__ import absolute_import, division, print_function

import torch.nn as nn


class Select(nn.Module):
    def __init__(self, select, validate=True):
        super().__init__()
        self.select = select
        if validate:
            self.validate()

    def extra_repr(self):
        return 'select=%r' % (self.select,)

    def validate(self):
        if isinstance(self.select, tuple):
            if len(self.select) > 1:
                assert self.select[1] == slice(None, None, None)
        else:
            assert not isinstance(self.select, list)

    def forward(self, x):
        return x[self.select]


class SelectChannel(Select):
    def validate(self):
        assert isinstance(self.select, tuple)
        assert len(self.select) > 1
        assert isinstance(self.select[1], slice)


class SelectChannelIndex(SelectChannel):
    def validate(self):
        assert isinstance(self.select, tuple)
        assert len(self.select) > 1
        assert isinstance(self.select[1], int)
