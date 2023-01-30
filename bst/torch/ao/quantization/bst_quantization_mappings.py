import torch
import torch.nn as nn

from bst.torch.ao.quantization import bst_quantable as bsnnq
from bst.torch.ao.quantization import bst_modules as bsnn

BST_QAT_MODULE_MAPPINGS = {
    nn.ReLU: bsnnq.ReLU,
    bsnn.Add: bsnnq.Add,
    bsnn.Mul: bsnnq.Mul,
    bsnn.Cat: bsnnq.Cat,
    bsnn.CatChannel: bsnnq.CatChannel,
    bsnn.A1000A0AvgPool2d: bsnnq.A1000A0AvgPool2d,
    nn.ConvTranspose2d: bsnnq.ConvTranspose2d
}

