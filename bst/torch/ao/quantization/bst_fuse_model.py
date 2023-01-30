import copy
import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from bst.torch.ao.quantization.bst_static_graph import QuantableGraph
from bst.torch.ao.quantization.bst_static_graph import rewrite_graph
from bst.torch.ao.quantization.bst_utils import non_empty

import bst.torch.ao.quantization as quantizer

logger = logging.getLogger(__name__)

FUSE_MODULE_PATTERN = [
 [nn.Conv1d, nn.BatchNorm1d],
 [nn.Conv2d, nn.BatchNorm2d],
 [nn.Conv3d, nn.BatchNorm3d],
 [nn.Linear, nn.BatchNorm1d],
 [nn.ConvTranspose1d, nn.BatchNorm1d],
 [nn.ConvTranspose2d, nn.BatchNorm2d],
 [nn.ConvTranspose3d, nn.BatchNorm3d]
]

class Forest(dict):
    def __init__(self, module_type=None, is_end=False):
        self.type = module_type
        self.children = {}
        self.is_end = is_end

def build_forest(forest, modules):
    prev = forest
    end = len(modules) - 1
    for i in range(len(modules)):
        cur = modules[i]
        if cur not in prev.children:
            prev.children[cur] = Forest(cur)
        
        prev.children[cur].is_end = i == end
        
        prev = prev.children[cur]

class FusableGraph(QuantableGraph):
    def __init__(self, model):
        super(FusableGraph, self).__init__(model)

    def pattern_matching(self, node, matched, lookup):
        # master list of final result, get updated on valid sequence matches
        matched_list = []

        target = node
        start = lookup
        tmp = []
        # look for node sequence matching one of patterns in lookup that have never been matched
        while self.uid_to_module[target].__class__ in start.children and target not in matched:
            tmp.append(target)

            # update start
            start = start.children[self.uid_to_module[target].__class__]

            # record the longest matching sequence
            if start.is_end:
                matched_list = tmp.copy()

            # update target search target
            if target not in self.children_nodes:
                break
            else:
                target = self.children_nodes[target][0]

        for matched_node in matched_list:
            matched.add(matched_node)

        return matched_list

    def get_fuse_list(self):
        # 0) parse FUSE_MODULE_PATTERN into faster lookup 
        lookup = Forest()

        for fuse_pattern in FUSE_MODULE_PATTERN:
            build_forest(lookup, fuse_pattern)

        # 1) traverse the graph and find pattern matching fuse module, if pattern found, add node names to fuse_list
        fuse_list = []

        # dfs to iterate over the graph and find matching fuse patterns,
        # a node can be visited not match or matched but not visited
        visited = set()
        matched = set()
        stack = []
        for node in self.nodes:
            if node not in visited:
                stack.append(node)

                while len(stack) > 0:
                    curr = stack.pop()

                    # find if there is matching module fuse pattern, if there is, add it to fuse_list
                    match = self.pattern_matching(curr, matched, lookup)
                    if match:
                        fuse_list.append(match)

                    visited.add(curr)

                    if curr in self.children_nodes:
                        for child_node in self.children_nodes[curr]:
                            if child_node not in visited:
                                stack.append(child_node)
        
        # 2) find torch module to module absolute name mapping
        def parse_module_name(module_name, module, name_mapping):
            if len(list(module.children())) == 0:
                name_mapping[module] = module_name

            for submodule_name, submodule in module.named_children():
                parse_module_name(module_name + '.' + submodule_name, submodule, name_mapping)

        module_mapping = {}
        for name, module in self.model.named_children():
            parse_module_name(name, module, module_mapping)

        # 3) use uid from fuse list and replaces it with absolute name
        for i in range(len(fuse_list)):
            for j in range(len(fuse_list[i])):
                module_uid = fuse_list[i][j]
                module = self.uid_to_module[module_uid]
                module_absolute_name = module_mapping[module]
                fuse_list[i][j] = module_absolute_name
        
        # 4) return name mapped fuse lists
        logger.info(f"modules to fuse: {fuse_list}")
    
        return fuse_list

def find_modules_to_fuse(model, input_tensor, debug_mode=False):
    assert input_tensor is not None, "Cannot find modules to fuse using None as input, input_tensor \
                is necessary to traverse the graph"
    model_copy = copy.deepcopy(model)
    
    gt = FusableGraph(model_copy)
    gt.parse(input_tensor)
    gt.build_connections()

    if debug_mode:
        gt.vis()
        gt.dot.render("static_graph_to_fuse.gv")

    return gt.get_fuse_list()

def fuse_model(model, inplace=False, input_tensor=None, debug_mode=False, **kwargs):
    # record model device
    original_device = 'cpu'
    if next(model.parameters()).is_cuda:
        original_device = 'cuda'

    # record model state
    state_is_training = False
    if model.training:
        state_is_training = True

    # change model mode to eval and device to cpu
    model.eval()
    model.to('cpu')
    if not inplace:
        model = copy.deepcopy(model)

    if input_tensor is not None and input_tensor.is_cuda:
        input_tensor = input_tensor.to('cpu')    
    
    modules_to_fuse = find_modules_to_fuse(model, input_tensor, debug_mode=debug_mode)

    logger.info(f"Modules to fuse: {modules_to_fuse}")    
    
    if 'fuse_custom_config_dict' in kwargs:
        user_fuse_custom_config_dict = kwargs['fuse_custom_config_dict']
        for k, v in user_fuse_custom_config_dict.items():
            fuse_custom_config_dict.update(k, v)
    else:
        fuse_custom_config_dict = None

    if non_empty(modules_to_fuse):
        quantizer.fuse_modules(model, modules_to_fuse, inplace=True, 
            fuse_custom_config_dict=fuse_custom_config_dict, **kwargs)

    # Split large Conv to small Convs based on BST hardware constrains
    if input_tensor is not None:
        rewrite_graph(model, input_tensor)

    # move model and input_tensor back to original device
    if original_device == 'cuda':
        model.to('cuda')
        if input_tensor is not None:
            input_tensor = input_tensor.to('cuda')

    # if is_training
    if state_is_training:
        model.train()
        
    return model