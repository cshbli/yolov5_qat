import torch
import torch.nn as nn

from .quantable_graph import QuantableGraph
from .rewrite_large_conv import rewrite_large_conv_matcher, rewrite_large_conv_rewriter


def get_strategies():
    strategies = [
        (rewrite_large_conv_matcher, rewrite_large_conv_rewriter)
    ]
    return strategies


class ModuleReplacement:

    def __init__(self, model):
        self.model = model

    def replace(self, input_tensor):
        assert self.model is not None and isinstance(self.model, nn.Module), "Expects model to be of type torch.nn.Module"

        # record model device
        original_device = 'cpu'
        if next(self.model.parameters()).is_cuda:
            original_device = 'cuda'
        
        # record model state
        state_is_training = False
        if self.model.training:
            state_is_training = True

        # change model mode to eval and device to cpu
        self.model.eval()
        self.model.to('cpu')
        if input_tensor.is_cuda:
            input_tensor = input_tensor.to('cpu')

        # CORE Logic:
        # build rewrite functions
        graph_rewrites = get_strategies()
        model_graph = QuantableGraph(self.model)
        model_graph.parse(input_tensor)
        model_graph.build_connections()
        model_graph.topology_sort()

        # a map to store the parent module of the current module, the current module is
        # a member of the named_children() of the parent module
        parent_module_map = {}
        def save_parent_module(model_graph, module):
            for _, submodule in module.named_children():
                parent_module_map[submodule] = module
                save_parent_module(model_graph, submodule)

        for name, module in self.model.named_children():
            save_parent_module(model_graph, module)

        def rewrite_subgraph(model_graph, module_name, module):
            matched = False
            for graph_rewrite in graph_rewrites:
                matcher, rewrite_func = graph_rewrite[0], graph_rewrite[1]
                match = matcher(model_graph, module)
                if match:
                    matched = True
                    rewrite_func(model_graph, module_name, match, parent_module_map)
                    break

            if not matched:
                for submodule_name, submodule in module.named_children():
                    rewrite_subgraph(model_graph, submodule_name, submodule)

        for name, module in self.model.named_children():
            rewrite_subgraph(model_graph, name, module)

        # move model and input_tensor back to original device
        if original_device == 'cuda':
            self.model.to('cuda')
            input_tensor = input_tensor.to('cuda')
        
        # if is_training
        if state_is_training:
            self.model.train()

    def validate(self):
        pass


def rewrite_graph(model, input_tensor):
    """
        Rewrite the graph inplace
    """
    module_replacer = ModuleReplacement(model)
    module_replacer.replace(input_tensor)
    module_replacer.validate()