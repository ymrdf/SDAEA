"""Frozen actor inference across independent robots using torch.func.vmap.

Weights are stacked once per frozen phase; each robot keeps independent state.
No critic/gradient computation is needed in this evaluation-only path.
"""
from collections import defaultdict
import numpy as np
import torch
from torch.func import functional_call, stack_module_state, vmap
from sdaea_streaming_v2 import State


def make_forward(model, recurrent):
    def forward(parameters, buffers, image, history, body, hidden, elapsed):
        state = State(image.unsqueeze(0), history.unsqueeze(0), body.unsqueeze(0), elapsed)
        if recurrent:
            output, nxt = functional_call(model, (parameters, buffers), (state, hidden.unsqueeze(0)))
            return output[0], nxt[0]
        output = functional_call(model, (parameters, buffers), (state,))
        return output[0], hidden
    return vmap(forward)


class FrozenPolicyBatch:
    def __init__(self, learners, variants, indices):
        self.groups = []
        self.locations = {}
        buckets = defaultdict(list)
        for i in indices:
            c = learners[i].c
            if not c.no_learn:
                raise ValueError('Frozen batching cannot update a training model')
            key = (c.width, c.depth, c.image_size, c.recurrent_size, c.recurrent_layer, c.cnn_extra)
            buckets[key].append(i)
        for indices in buckets.values():
            models = [learners[i].actor for i in indices]
            parameters, buffers = stack_module_state(models)
            for value in parameters.values():
                value.requires_grad_(False)
            sample = next(models[0].parameters())
            size = learners[indices[0]].c.recurrent_size
            group = dict(indices=indices, parameters=parameters, buffers=buffers,
                         forward=make_forward(models[0], bool(size)),
                         hidden=sample.new_zeros(len(indices), size or 1),
                         exploration=sample.new_tensor([learners[i].c.exploration for i in indices]).unsqueeze(1),
                         clear=[j for j,i in enumerate(indices) if variants[i].get('clear_memory')])
            self.groups.append(group)
            for j,i in enumerate(indices):
                self.locations[i]=(group,j)

    @torch.no_grad()
    def probabilities(self, states):
        result = {}
        for g in self.groups:
            if g['clear']:
                g['hidden'][g['clear']] = 0
            sequence = [states[i] for i in g['indices']]
            logits, g['hidden'] = g['forward'](
                g['parameters'], g['buffers'],
                torch.cat([s.image for s in sequence]),
                torch.cat([s.history for s in sequence]),
                torch.cat([s.body for s in sequence]),
                g['hidden'], g['hidden'].new_tensor([s.elapsed for s in sequence]))
            probabilities = torch.softmax(logits, -1)
            probabilities = (1-g['exploration'])*probabilities + g['exploration']/probabilities.shape[-1]
            values=probabilities.cpu().numpy().astype(float)
            for i,p in zip(g['indices'], values):
                result[i]=p/p.sum()
        return result

    @torch.no_grad()
    def clear_indices(self, indices):
        for i in indices:
            if i in self.locations:
                group,j=self.locations[i]
                group['hidden'][j].zero_()
