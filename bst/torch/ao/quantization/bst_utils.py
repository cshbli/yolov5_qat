import copy
import numpy as np
from typing import List, Tuple, Dict
from collections import OrderedDict

import torch
from bst.torch.ao.quantization.quantization_mappings import DEFAULT_QAT_MODULE_MAPPINGS, DEFAULT_STATIC_QUANT_MODULE_MAPPINGS

from bst.torch.ao.quantization.bst_quantization_mappings import BST_QAT_MODULE_MAPPINGS
# from bst.torch.ao.quantization.bst_quantable_modules.quantable_module import QuantableModule

"""
    The BST custom module modified original torch module's functionality with hardware constraints.
    When the module is exported, the module must be converted to torch's original module because
    there is no ONNX op associated with bst's custom module. Therefore, we must inplace replace 
    Torch model's custom ops with original torch supported ops
"""
def replace_custom_ops(model):
    for child_name, child in model.named_children():
        # if not isinstance(child, QuantableModule) and hasattr(child, '_ORIGIN_MODULE'):
        #     setattr(model, child_name, child.to_origin(**child.kwargs))
        replace_custom_ops(child)

def is_quantable_module(module):
    default_quantable_modules = set(copy.deepcopy(DEFAULT_QAT_MODULE_MAPPINGS).values())
    default_quantable_modules.update(set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.values()))

    quantable_stub_mapping = OrderedDict(
        quanted_input=torch.quantization.QuantStub,
        dequanted_output=torch.quantization.DeQuantStub,
    )
    default_quantable_modules.update(set(BST_QAT_MODULE_MAPPINGS.values()))
    default_quantable_modules.update(set(quantable_stub_mapping.values()))
    
    return isinstance(module, tuple(default_quantable_modules))

# TODO: move to graph class
def is_quantable(graph, node_name):
    joint_qat_module_mapping = copy.deepcopy(DEFAULT_QAT_MODULE_MAPPINGS)
    joint_qat_module_mapping.update(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS)
    quantable_stub_mapping = OrderedDict(
        quanted_input=torch.quantization.QuantStub,
        dequanted_output=torch.quantization.DeQuantStub,
    )
    joint_qat_module_mapping.update(BST_QAT_MODULE_MAPPINGS)
    joint_qat_module_mapping.update(quantable_stub_mapping)
    module = graph.uid_to_module[node_name]


    return type(module) in joint_qat_module_mapping.values() or type(module) in joint_qat_module_mapping

'''
    Find the cloest quantable parent. If the current node is quantable, return itself
'''
# TODO: move to graph class
def get_quantable_node(graph, node_name):
    joint_qat_module_mapping = copy.deepcopy(DEFAULT_QAT_MODULE_MAPPINGS)
    joint_qat_module_mapping.update(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS)
    quantable_stub_mapping = OrderedDict(
        quanted_input=torch.quantization.QuantStub,
        dequanted_output=torch.quantization.DeQuantStub,
    )
    joint_qat_module_mapping.update(BST_QAT_MODULE_MAPPINGS)
    joint_qat_module_mapping.update(quantable_stub_mapping)

    module = graph.uid_to_module[node_name]
    if type(module) in joint_qat_module_mapping or type(module) in joint_qat_module_mapping.values():
        return node_name
    else:
        return get_parent_quantable(graph, node_name)

'''
    Find the cloest quantable parent that's not the current node.
'''
# TODO: move to graph class
def get_parent_quantable(graph, node_name):
    joint_qat_module_mapping = copy.deepcopy(DEFAULT_QAT_MODULE_MAPPINGS)
    joint_qat_module_mapping.update(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS)
    quantable_stub_mapping = OrderedDict(
        quanted_input=torch.quantization.QuantStub,
        dequanted_output=torch.quantization.DeQuantStub,
    )
    joint_qat_module_mapping.update(BST_QAT_MODULE_MAPPINGS)
    joint_qat_module_mapping.update(quantable_stub_mapping)

    module = graph.uid_to_module[node_name]

    # find quantable parent
    for parent in graph.parent_nodes[node_name]:
        ref_param_entry = None
        
        while not ref_param_entry:
            parent_module = graph.uid_to_module[parent]
            # TODO: use qat module full mapping

            # keep looking for parent nodes if current parent is not a quantable node
            if type(parent_module) not in joint_qat_module_mapping and type(parent_module) not in joint_qat_module_mapping.values():
                parent = graph.parent_nodes[parent][0]
                continue
            
            ref_param_entry = parent
    
        return ref_param_entry
    return None

'''
    Find the all closest quantable parent that's not the current node.
'''
# TODO: move to graph class
def get_parent_quantables(graph, node_name):
    joint_qat_module_mapping = copy.deepcopy(DEFAULT_QAT_MODULE_MAPPINGS)
    joint_qat_module_mapping.update(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS)
    quantable_stub_mapping = OrderedDict(
        quanted_input=torch.quantization.QuantStub,
        dequanted_output=torch.quantization.DeQuantStub,
    )
    joint_qat_module_mapping.update(BST_QAT_MODULE_MAPPINGS)
    joint_qat_module_mapping.update(quantable_stub_mapping)

    module = graph.uid_to_module[node_name]

    # find quantable parent
    quantable_parents = set()
    for parent in graph.parent_nodes[node_name]:
        ref_param_entry = None
        
        while not ref_param_entry:
            parent_module = graph.uid_to_module[parent]
            # TODO: use qat module full mapping

            # keep looking for parent nodes if current parent is not a quantable node
            if type(parent_module) not in joint_qat_module_mapping and type(parent_module) not in joint_qat_module_mapping.values():
                parent = graph.parent_nodes[parent][0]
                continue
            
            ref_param_entry = parent
    
        if ref_param_entry is not None:
            quantable_parents.add(ref_param_entry)
            
    return list(quantable_parents)

def calculate_quant_scale_zp(rmin, rmax, qmin, qmax, symmetric=True, power_of_two_scale=True):
    '''
    Calculate the scale s and zero point z for the quantization relation
    r = s(q-z), where r are the original values and q are the corresponding
    quantized values.
    r and z are calculated such that every value within [rmin,rmax] has an
    approximate representation within [qmin,qmax]. In addition, qmin <= z <=
    qmax is enforced. If the symmetric flag is set to True, the interval
    [rmin,rmax] is symmetrized to [-absmax, +absmax], where
    absmax = max(abs(rmin), abs(rmax)).
    power_of_two_scale: True or False to constraint rmin, rmax
    :parameter rmin: minimum value of r
    :parameter rmax: maximum value of r
    :parameter qmin: minimum value representable by the target quantization data type
    :parameter qmax: maximum value representable by the target quantization data type
    :return: scale and zero - s, z
    '''
    # Adjust rmin and rmax such that 0 is included in the range. This is
    # required to make sure zero can be represented by the quantization data
    # type (i.e. to make sure qmin <= zero_point <= qmax)
    # rmin = min(rmin, 0)
    # rmax = max(rmax, 0)

    if rmin == rmax == 0:
        rmin = -1
        rmax = 1

    if symmetric:
        absmax = max(abs(rmin), abs(rmax))
        rmin = - absmax
        rmax = + absmax
        abs_qmax = max(abs(qmin), abs(qmax))
        qmin = - abs_qmax
        qmax = abs_qmax

    if power_of_two_scale:
        rmin = np.power(2, np.ceil(np.log2(abs(rmin)))) * np.sign(rmin)
        rmax = np.power(2, np.ceil(np.log2(abs(rmax)))) * np.sign(rmax)

    scale = (rmax - rmin) / float(qmax - qmin) if rmax != rmin else 1.0
    zero_point = round(qmin - rmin / scale)

    return scale, zero_point

def non_empty(data):
    if data is None:
        return False
    
    result = False
    if isinstance(data, (List, Tuple)):
        for entry in data:
            result = result or non_empty(entry)
    elif isinstance(data, Dict):
        result = True if len(data) > 0 else False
    else:
        result = True
    
    return result