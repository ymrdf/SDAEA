"""Transfer an existing plain visual network into wider/deeper plain networks.

Uniform neuron replication preserves a same-depth network before small symmetry
breaking noise. Extra normalized LeakyReLU blocks are identity-initialized but
are NOT function-preserving; baseline evaluation measures this initial change.
"""
import math
import torch
from torch import nn

@torch.no_grad()
def grow_network(target, source_state, source_width, source_depth, noise=1e-4):
    width=target.output.in_features
    if width % source_width or len(target.hidden)//3 < source_depth:
        raise ValueError('Require integer width expansion and nondecreasing depth')
    factor=width//source_width
    for key,value in target.conv.state_dict().items():
        value.copy_(source_state['conv.'+key])
    for index in range(len(target.hidden)//3):
        layer=target.hidden[index*3]
        if index<source_depth:
            weight=source_state[f'hidden.{index*3}.weight']
            layer.weight.copy_(weight.repeat(factor,1) if index==0 else weight.repeat(factor,factor)/factor)
            layer.bias.copy_(source_state[f'hidden.{index*3}.bias'].repeat(factor))
        else:
            nn.init.eye_(layer.weight);nn.init.zeros_(layer.bias)
        if noise:
            layer.weight.add_(torch.randn_like(layer.weight),alpha=noise/math.sqrt(layer.in_features))
    target.output.weight.copy_(source_state['output.weight'].repeat(1,factor)/factor)
    target.output.bias.copy_(source_state['output.bias'])
