from __future__ import absolute_import, division, print_function

import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *args): 
        super(Reshape, self).__init__() 
        self.shape = args 
    
    def forward(self, x): return x.view(self.shape)

class ReshapeRT(nn.Module):
    '''
    Reshape nn.Module supporting flexible shape during caller forward function
    '''
    def __init__(self): 
        super(ReshapeRT, self).__init__()
    
    def forward(self, x, shape):
        return x.view(shape)


class ReshapeChannel(Reshape):
    pass


class View(nn.Module):
    def forward(self, x, shape):
        return x.view(shape)


class ViewChannel(View):
    pass
