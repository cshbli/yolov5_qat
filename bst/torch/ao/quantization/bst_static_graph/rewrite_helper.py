import copy
import numpy as np

import torch
import torch.nn as nn

from bst.torch.ao.quantization import bst_modules as bstnn


def gen_split_conv(module, activation):
    '''
        Given a conv module and its activation, this function split the conv module 
        provided by number of splits.

                Input
                  |
                Conv
                  |
                Relu
                  |
                Output
        =>
                Input
              /   | ...\
            Conv  Conv  Conv
             |    | ... |
            Relu  Relu  Relu
              \   | .../
                Concat
                  |
                Output
    '''
    KB = 1024
    weights = module.weight
    weight_size = [*weights.size()]
    unit_size = np.prod([x for i, x in enumerate(weight_size) if i != 1])
    # split on output channel, weight shape: (Cout, Cin, HH, WW)
    total_channels = weight_size[0]

    # binary search to find largest channel size that is less than max size 
    def get_split_channel(c, target):
        s, e = 0, c
        while s < e:
            mid = s + (e - s) // 2
            if mid * unit_size == target:
                return mid
            elif mid * unit_size > target:
                e = mid
            else:
                s = mid + 1

        return mid

    # find number number of splits for large weight channel, split is done on output channel
    # split method is even split
    splits = []
    target_max_size = 2.0 * KB * KB
    while total_channels > 0:
        if target_max_size > total_channels * unit_size:
            splits.append(total_channels)
            total_channels = 0
        else:
            split_channel = get_split_channel(total_channels, target_max_size)
            splits.append(split_channel)
            total_channels -= split_channel
    
    # build new module and add it to sequential
    new_module = bstnn.SplitConv(module, activation, splits)

    return new_module


def gen_identity():
    return nn.Identity()