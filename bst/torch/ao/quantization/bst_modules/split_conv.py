import copy

import torch
import torch.nn as nn

from .cat import CatChannel

class SplitBranch(nn.Module):
    def __init__(self, in_channels, split, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode, activation):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, split, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode)

        self.act = None
        if activation is not None:
            self.act = copy.deepcopy(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x

class SplitConv(nn.Module):
    def __init__(self, module, activation, splits):
        super().__init__()

        # record original parameters
        in_channels = module.in_channels
        kernel_size = module.kernel_size
        
        stride = module.stride
        padding = module.padding
        dilation = module.dilation
        groups = module.groups
        use_bias = module.bias is not None
        padding_mode = module.padding_mode

        # build modules
        self.module_list = []
        start = 0
        for i in range(len(splits)):
            split = splits[i]
            module_name = 'split_' + str(i)
            branch_module = SplitBranch(in_channels, split, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode, activation)

            # set weights and bias based on split channels
            new_weights = nn.Parameter(module.weight[start : start + split,:,:,:].clone())
            branch_module.conv.weight = new_weights
            if use_bias:
                new_bias =  nn.Parameter(module.bias[start : start + split].clone())
                branch_module.conv.bias = new_bias
            
            start += split

            setattr(self, module_name, branch_module)
            self.module_list.append(branch_module)

        self.concat = CatChannel()
        
    def forward(self, x):
        branch_out = []
        for seq in self.module_list:
            branch_out.append(seq(x))
        
        return self.concat(*branch_out)