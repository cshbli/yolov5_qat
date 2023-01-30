import logging

from graphviz import Digraph
from collections import OrderedDict
from typing import (List, Tuple, Dict)

import torch
import torch.nn as nn

from bst.torch.ao.quantization.fake_quantize import FakeQuantizeBase

logger = logging.getLogger(__name__)


"""
    Node class for graph:

    Current:
        Node(
            module,
            inputs = List(Node),
            outputs = List(Node),
            name
        )

    Ultimate:
        Node(
            container(nn.Module | nn.Functional | Method),
            inputs = List(Node),
            outputs = List(Node),
            name
        )

    The node class captures vertices in torch's static graph. Since torch's computation can take
    the form of nn.Module, nn.Functional, and method. The Node should be able to wrap each of these
    computation blocks and create a static graph. Currently, it's only able to capture modules. Thus, 
    the input torch model can only contain nn.Modules. Otherwise the graph will be incorrectly parsed
"""

class Node(object):
    """ abstract class that holds nodes and edges"""
    
    def __init__(self, module):
        self.module = module
        self.inputs = []
        self.outputs = []
        self.name = '-'.join([module.__class__.__name__, str(id(module))])
        
    def add_input(self, node):
        if node not in self.inputs:
            self.inputs.append(node)
    
    def add_output(self, node):
        if node not in self.outputs:
            self.outputs.append(node)

"""
    The static graph view of torch's dynamic graph. It is created to be easily changable when locations
    of modules are known during computation. Static graph are easier to manipulated and hence more perfered
    during model optimization/rewrites.
"""

class QuantableGraph(object):

    def __init__(self, model):
        self.model = model

        self.collection = []
        self.name_mapping = OrderedDict()

        for submodule_name, submodule in self.model.named_children():
            self._traverse(submodule, submodule_name)

    def _traverse(self, module, module_name=""):
        """ Global searching individual modules """

        # add to graph node collection if the module has no more children, meaning its a leaf module
        # or (in Fake Quantized Model) the children has FakeQuantize
        if len(list(module.children())) == 0 and not isinstance(module, nn.Identity):
            self.collection.append(module)
            self.name_mapping[module] = module_name
        else:
            has_fake_quant = False
            for m_child in list(module.children()):
                if isinstance(m_child, FakeQuantizeBase):
                    has_fake_quant = True
                    break
            
            if has_fake_quant or isinstance(module, nn.quantized.FloatFunctional):
                self.collection.append(module)
                self.name_mapping[module] = module_name
            else:
                for submodule_name, submodule in module.named_children():
                    self._traverse(submodule, module_name + '.' + submodule_name)

    def parse(self, input_tensor):
        """ Parse topology of model with dummpy input. """
        _ = self.model.cpu().eval()

        # TODO: set module inplace setting to False and reenble it once we 
        # are done. Problem with inplace ops is that the data address if its
        # input and output tensors are the same, which creates problem when
        # we need to construct a graph, we will have no idea how to connect
        # two modules if there are any inplace tensors from auto grad
        
        # mapping between fake quantize to float functionals
        fq_to_ff = OrderedDict()
        for m in self.collection:
            if isinstance(m, nn.quantized.FloatFunctional) and hasattr(m, 'activation_post_process'):
                fq_to_ff[m.activation_post_process] = m
        
        ## 1. find out linked relationship between modules
        ## save tensor so that it will not be recycled
        saved_tensors = set()
        saved_grad_func = set()
        tensor_to_module = OrderedDict()
        module_to_tensor = OrderedDict()
        gradfn_to_tensor = OrderedDict()

        # dfs function to search parent module for inputs tensors fed into nn.quantized.FloatFunctionals
        def gradfn_dfs(fn):
            if not fn or not hasattr(fn, 'next_functions'):
                return []
            results = []
            if fn in gradfn_to_tensor:
                assert fn in gradfn_to_tensor, "Parent output tensor's gradient function is not recorded"
                fn_tensor = gradfn_to_tensor[fn]
                results.append(fn_tensor)
                return results

            for next_gradfn in fn.next_functions:
                results.extend(gradfn_dfs(next_gradfn[0]))
            
            return results

        # saves: {tensor: module}
        # saves: {tensor._base.grad_fn: tensor}
        def memorize_input_tensor(_input, saved_tensors, tensor_to_module, 
                                    true_module, saved_grad_func, gradfn_to_tensor):
            saved_tensors.add(_input)
            tensor_to_module.setdefault(_input.storage().data_ptr(), []).append(true_module)
            if not _input.grad_fn:
                return
            else:
                saved_grad_func.add(_input.grad_fn)
                gradfn_to_tensor[_input.grad_fn] = _input

            if not hasattr(_input, '_base'):
                return
            elif _input._base is not None:
                saved_grad_func.add(_input._base.grad_fn)
                gradfn_to_tensor[_input._base.grad_fn] = _input

        # saves: {module: tensor}
        # saves: {tensor._base.grad_fn: tensor}
        def memorize_output_tensor(_output, saved_tensors, module_to_tensor, 
                                    true_module, saved_grad_func, gradfn_to_tensor):
            saved_tensors.add(_output)
            module_to_tensor.setdefault(true_module, []).append(_output.storage().data_ptr())
            if not _output.grad_fn:
                return
            else:
                saved_grad_func.add(_output.grad_fn)
                gradfn_to_tensor[_output.grad_fn] = _output

            if not hasattr(_output, '_base'):
                return
            elif _output._base is not None:
                saved_grad_func.add(_output._base.grad_fn)
                gradfn_to_tensor[_output._base.grad_fn] = _output

        def forward_hook_(module, inputs, outputs):
            true_inputs = []
            true_outputs = []
            true_module = module

            if module in fq_to_ff:
                # look for fake quant's input modules
                true_module = fq_to_ff[module]

                found_tensors = set()
                for _input in inputs:
                    _input_gradfn = _input.grad_fn

                    # check grad func with dfs and search for all tensors with modules in module_to_tensor map
                    found_tensors.update(gradfn_dfs(_input_gradfn))
                
                true_inputs = list(found_tensors)
                true_outputs = outputs
            else:
                true_inputs = inputs
                true_outputs = outputs

            # save input and output tensor to mappings for easy search
            for _input in true_inputs:
                if isinstance(_input, List):
                    for _l_input in _input:
                        memorize_input_tensor(_l_input, saved_tensors, tensor_to_module, 
                                    true_module, saved_grad_func, gradfn_to_tensor)
                else:
                    memorize_input_tensor(_input, saved_tensors, tensor_to_module, 
                                true_module, saved_grad_func, gradfn_to_tensor)

            for _output in true_outputs:
                if isinstance(_output, List):
                    for _o_input in _output:
                        memorize_output_tensor(_o_input, saved_tensors, module_to_tensor, 
                                    true_module, saved_grad_func, gradfn_to_tensor)
                else:
                    memorize_output_tensor(_output, saved_tensors, module_to_tensor, 
                                true_module, saved_grad_func, gradfn_to_tensor)

        hook_handlers_ = [ m.register_forward_hook(forward_hook_) for m in self.collection]

        # register forward hook for nn.quantized.FloatFunctional's activation post process
        for m in self.collection:
            if isinstance(m, nn.quantized.FloatFunctional) \
                and hasattr(m, 'activation_post_process'):
                hook_handlers_.append(m.activation_post_process.register_forward_hook(forward_hook_))

        _ = self.model(input_tensor)
        
        ## clean registered hook for recording graph topology and save tensors
        for handler in hook_handlers_:
            handler.remove()
        saved_tensors.clear()

        def _build_graph(tensor, parent=None):
            modules = tensor_to_module.get(tensor, None)
            # if module is leaf then we don't need to do anything
            if modules is None:
                logger.info(f"Tensor: {tensor} does not have succeeding child module, it's parent module is {parent}")
                return []

            node_list = []
            for module in modules:
                # if module has been visited, we just need to update its input tensor
                # if module has not been visited, we need to create its node, add its input tensor, 
                #   and visit its children
                if module in module_to_node:
                    node = module_to_node[module]

                    # create placeholder for real graph inputs if tensor is one of graph's input tensors
                    if parent and parent.module != module:
                        node.inputs.append(parent)
                    node_list.append(node)
                    continue
                elif module not in module_to_node:
                    node = Node(module)
                    module_to_node[module] = node

                    # create placeholder for real graph inputs if tensor is one of graph's input tensors
                    if parent and parent.module != module:
                        node.inputs.append(parent)
                for _output_tensor in module_to_tensor[module]:
                    outputs = _build_graph(_output_tensor, node)
                    for _output in outputs:
                        if _output is not node:
                            node.outputs.append(_output)
                node_list.append(node)
                
            return node_list
            
        ## use input_tensor, tensor_to_module, and module_to_tensor to build topology graph
        module_to_node = OrderedDict()

        root_nodes = []
        for _input_tensor in input_tensor:
            root_nodes.extend(_build_graph(_input_tensor.storage().data_ptr(), None))
        
        self.root_nodes = root_nodes
        self.module_to_node = module_to_node
    
    def build_connections(self):
        # STEP1: build graph relationship and keep track of parent and children nodes
        keys = list(self.module_to_node.keys())
        self.uid_to_module = {}
        
        self.nodes = []
        self.parent_nodes = {}
        self.children_nodes = {}
        
        for key in keys:
            nodename = '-'.join([key.__class__.__name__, str(id(key))])
            self.uid_to_module[nodename] = key
            self.nodes.append(nodename)
        
        visited_edges = set()
        for key in keys:
            nodename = '-'.join([key.__class__.__name__, str(id(key))])
            output_nodes = self.module_to_node[key].outputs
            input_nodes = self.module_to_node[key].inputs
            for node in output_nodes:
                output_node_name = '-'.join([node.module.__class__.__name__, str(id(node.module))])
                if (nodename, output_node_name) not in visited_edges:
                    visited_edges.add((nodename, output_node_name))
                    
                    # add to children nodes
                    if nodename not in self.children_nodes:
                        self.children_nodes[nodename] = [output_node_name]
                    else:
                        self.children_nodes[nodename].append(output_node_name)
                    
                    # add to parent nodes
                    if output_node_name not in self.parent_nodes:
                        self.parent_nodes[output_node_name] = [nodename]
                    else:
                        self.parent_nodes[output_node_name].append(nodename)

            for node in input_nodes:
                input_node_name = '-'.join([node.module.__class__.__name__, str(id(node.module))])
                if (input_node_name, nodename) not in visited_edges:
                    visited_edges.add((input_node_name, nodename))

                    # add to children nodes
                    if input_node_name not in self.children_nodes:
                        self.children_nodes[input_node_name] = [nodename]
                    else:
                        self.children_nodes[input_node_name].append(nodename)
                    
                    # add to parent nodes
                    if nodename not in self.parent_nodes:
                        self.parent_nodes[nodename] = [input_node_name]
                    else:
                        self.parent_nodes[nodename].append(input_node_name)

    def _topology_sort_util(self, node_name: str, node_execute_list: list, visited: dict):
        visited[node_name] = 1
        if node_name not in self.children_nodes:
            node_execute_list.append(node_name)
            return
        
        child_node_names = self.children_nodes[node_name][::-1]
        for child_node_name in child_node_names:
            if child_node_name in visited:
                continue
            self._topology_sort_util(child_node_name, node_execute_list, visited)
        node_execute_list.append(node_name)

    def topology_sort(self):
        node_execute_list = []
        visited = {}
        for node_name in self.nodes:
            if node_name in visited:
                continue
            self._topology_sort_util(node_name, node_execute_list, visited)
        node_execute_list = node_execute_list[::-1]
        self.nodes = node_execute_list

    def vis(self):
        """ Visualize the topology with Graphviz tool. """
        
        if 'module_to_node' not in dir(self):
            raise ValueError('Run parse() method before visualize.')
            
        node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='10',
                     ranksep='0.1',
                     height='0.2',
                     fontname='monospace')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        
        keys = list(self.module_to_node.keys())
        
        self.uid_to_module = {}
        
        for key in keys:
            nodename = '-'.join([key.__class__.__name__, str(id(key))])
            dot.node( nodename )
            self.uid_to_module[nodename] = key
            
        visited_edges = set()
        for key in keys:
            nodename = '-'.join([key.__class__.__name__, str(id(key))])
            output_nodes = self.module_to_node[key].outputs
            input_nodes = self.module_to_node[key].inputs
            for node in output_nodes:
                output_node_name = '-'.join([node.module.__class__.__name__, str(id(node.module))])
                if (nodename, output_node_name) not in visited_edges:
                    visited_edges.add( (nodename, output_node_name) )
                    dot.edge( nodename, output_node_name )

            for node in input_nodes:
                input_node_name = '-'.join([node.module.__class__.__name__, str(id(node.module))])
                if (input_node_name, nodename) not in visited_edges:
                    dot.edge( input_node_name, nodename )
                    visited_edges.add( (input_node_name, nodename) )
        self.dot = dot          
