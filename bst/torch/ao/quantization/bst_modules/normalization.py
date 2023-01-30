from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class Normalization(nn.Module):
    def __init__(self, data_dims, weight=None, bias=None):
        super().__init__()
        if weight is not None:
            if isinstance(weight, nn.Parameter):
                self.weight = weight
            else:
                weight = torch.as_tensor(weight)
                self.register_buffer('weight', weight)
            self.weight_shape = (1, *weight.shape, *([1] * (data_dims - weight.dim() - 1)))
        else:
            self.register_buffer('weight', None)
        if bias is not None:
            if isinstance(bias, nn.Parameter):
                self.bias = bias
            else:
                bias = torch.as_tensor(bias)
                self.register_buffer('bias', bias)
            self.bias_shape = (1, *bias.shape, *([1] * (data_dims - bias.dim() - 1)))
        else:
            self.register_buffer('bias', None)

    def extra_repr(self):
        return 'weight=%s,bias=%s' % (
            self.weight.cpu().numpy() if self.weight is not None else None,
            self.bias.cpu().numpy() if self.bias is not None else None
        )

    def forward(self, input):
        weight = self.weight
        if weight is not None:
            input = input * weight.reshape(*self.weight_shape)
        bias = self.bias
        if bias is not None:
            input = input + bias.reshape(*self.bias_shape)
        return input

    @staticmethod
    def onnx_nodes(module, module_name, nodes, index, module_parsers, forward=False):
        assert isinstance(module, Normalization), "module %s type %s is not Normalization" % (
            module_name, type(module))
        return module.get_onnx_nodes(module_name, nodes, index, module_parsers, forward=forward)

    def get_onnx_nodes(self, module_name, nodes, index, module_parsers, forward=False):
        onnx_nodes = [
            dict(
                node_name="%s.mul" % module_name,
                patterns=None, input_indexes=[], output_indexes=[]
            )
        ]
        bias = self.bias
        if bias is not None:
            onnx_nodes.append(
                dict(
                    node_name="%s.add" % module_name,
                    patterns=None, input_indexes=[], output_indexes=[]
                )
            )
        onnx_nodes[0]['input_indexes'] = None
        onnx_nodes[-1]['node_name'] = module_name
        onnx_nodes[-1]['output_indexes'] = None
        return onnx_nodes
