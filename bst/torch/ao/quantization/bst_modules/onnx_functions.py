from __future__ import absolute_import, division, print_function

import torch.nn as nn

from bstnnx_training.PyTorch.QAT.modules import interpolate


def pooling_onnx_nodes(module, module_name, nodes, index, module_parsers, forward=False):
    onnx_nodes = []

    # match generated pad operator if pad is not 0 or tuples of 0
    pooling_node = nodes[index]
    if pooling_node['op_type'] == 'AveragePool':
        pad_val = module.padding
        if not isinstance(pad_val, tuple) and module.padding != 0:
            onnx_nodes.append(dict(
                node_name='%s.pad' % module_name,
                patterns=None, input_indexes=[], output_indexes=[]
            ))
        elif isinstance(pad_val, tuple):
            pad_val = list(set(module.padding))
            if len(pad_val) != 1 or pad_val[0] != 0:
                onnx_nodes.append(dict(
                    node_name='%s.pad' % module_name,
                    patterns=None, input_indexes=[], output_indexes=[]
                ))
    
    if pooling_node['op_type'] == 'AveragePool' or pooling_node['op_type'] == 'MaxPool':
        onnx_nodes.append(dict(
            node_name=module_name,
            patterns=None, input_indexes=[], output_indexes=[]
        ))

    onnx_nodes[0]['input_indexes'] = None
    onnx_nodes[-1]['node_name'] = module_name
    onnx_nodes[-1]['output_indexes'] = None
    return onnx_nodes


def linear_onnx_nodes(module, module_name, nodes, index, module_parsers, forward=False):
    assert isinstance(module, nn.Linear), "module %s type is not Linear: %s" % (
        module_name, type(module))
    onnx_nodes = []
    gemm_node = nodes[index]
    if gemm_node['op_type'] == 'Gemm':
        onnx_nodes.append(dict(
            node_name=module_name,
            patterns=None, input_indexes=[], output_indexes=[]
        ))
    else:
        onnx_nodes.append(dict(
            node_name='%s.mul' % module_name,
            patterns=None, input_indexes=[], output_indexes=[]
        ))
        if module.bias is not None:
            onnx_nodes.append(dict(
                node_name='%s.add' % module_name,
                patterns=None, input_indexes=[], output_indexes=[]
            ))
    onnx_nodes[0]['input_indexes'] = None
    onnx_nodes[0]['node_name'] = module_name
    onnx_nodes[-1]['output_indexes'] = None
    return onnx_nodes


def interpolate_onnx_nodes(module, module_name, nodes, index, module_parsers, forward=False):
    assert isinstance(module, (nn.Upsample, interpolate.Interpolate)), \
        "module %s type is not Upsample or Interpolate: %s" % (module_name, type(module))
    onnx_nodes = [
        dict(
            node_name=module_name,
            patterns=None, input_indexes=[], output_indexes=[]
        )
    ]
    if module.size is not None:
        if forward:
            while index < len(nodes):
                index += 1
                onnx_nodes = onnx_nodes + [dict(
                    node_name=None, patterns=None,
                    input_indexes=[], output_indexes=[]
                )]
                if nodes[index]['op_type'] == 'Resize':
                    break
        else:
            while index > 0:
                index -= 1
                onnx_nodes = [dict(
                    node_name=None, patterns=None,
                    input_indexes=[], output_indexes=[]
                )] + onnx_nodes
                if nodes[index]['op_type'] == 'Slice':
                    break
    onnx_nodes[-1]['input_indexes'] = None
    onnx_nodes[-1]['output_indexes'] = None
    return onnx_nodes


def create_onnx_nodes_by_size(onnx_nodes_size):
    onnx_nodes = []
    for i in range(onnx_nodes_size):
        onnx_nodes.append(dict(
            node_name=None,
            patterns=None, input_indexes=[], output_indexes=[]
        ))
    onnx_nodes[0]['input_indexes'] = None
    onnx_nodes[0]['node_name'] = '%(module_name)s'
    onnx_nodes[-1]['output_indexes'] = None
    return onnx_nodes


def onnx_nodes_until(op_type, include=False):
    def get_onnx_nodes(module, module_name, nodes, index, module_parsers, forward=False):
        current_index = index
        onnx_nodes_size = 0
        if forward:
            nodes_size = len(nodes)
            while current_index < nodes_size:
                if nodes[current_index]['op_type'] == op_type:
                    break
                current_index += 1
            if current_index == nodes_size:
                raise Exception("Does not find op_type %s forward from index %s to end" %
                                (op_type, index))
            onnx_nodes_size = current_index - index
        else:
            while current_index >= 0:
                if nodes[current_index]['op_type'] == op_type:
                    break
                current_index -= 1
            if current_index < 0:
                raise Exception(
                    "Does not find opt_type %s backward from index %s to beginning" % (op_type, index))
            onnx_nodes_size = index - current_index
        if include:
            onnx_nodes_size += 1
        return create_onnx_nodes_by_size(onnx_nodes_size)

    return get_onnx_nodes
