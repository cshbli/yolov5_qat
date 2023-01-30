import math
import numpy as np

import torch
import torch.nn as nn

import logging

from bst.torch.ao.quantization.bst_static_graph import rewrite_helper

logger = logging.getLogger(__name__)

CONV_TYPE = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d)

ACTIVATION_TYPE = (
    nn.ReLU, nn.LeakyReLU)

KB = 1024
MAX_WEIGHT_SIZE = 2.0 * KB * KB

def rewrite_large_conv_matcher(model_graph, module):
    match = []

    # Rule 1: module is conv type
    if not isinstance(module, CONV_TYPE):
        return []
    
    # Rule 2: module has weight
    if not  hasattr(module, 'weight'):
        return []
    
    # Rule 3: module weight total is larger than max size
    weights = module.weight
    weight_size = np.prod([*weights.size()])

    if weight_size > MAX_WEIGHT_SIZE:
        match.append(module)
    else:
        return []

    # Find conv node and activation if any
    node = model_graph.module_to_node[module]
    children = model_graph.children_nodes[node.name]
    if children:
        if len(children) == 1:
            child_uid = children[0]
            child_module = model_graph.uid_to_module[child_uid]
            if isinstance(child_module, ACTIVATION_TYPE):
                match.append(child_module)
        else:
            match.append(None)
    else:
        match.append(None)
    
    return match


def rewrite_large_conv_rewriter(model_graph, module_name, match, parent_module_map):
    logger.debug(f"rewrite_large_conv, match: {match}")

    original_module = match[0]
    assert original_module in parent_module_map, f"Error, {type(original_module)} is not wrapped by a parent"

    # replace conv module with split conv module
    activation = None
    if len(match) > 1 and match[1]:
        activation = match[1]
        assert activation in parent_module_map, f"Error, {type(activation)} is not wrapped by a parent"
    org_module_parent_module = parent_module_map[original_module]
    setattr(org_module_parent_module, module_name, rewrite_helper.gen_split_conv(original_module, activation))

    # set activation layer to identity since it is added by gen_split_conv() function
    if activation:
        activation_name = None
        activation_parent = parent_module_map[activation]
        for child_name, child_module in activation_parent.named_children():
            if child_module == activation:
                activation_name = child_name
                break

        assert activation_name is not None, "Error: Cannot find activation's layer name in parent module"
        setattr(activation_parent, activation_name, rewrite_helper.gen_identity())