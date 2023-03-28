import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from contextlib import contextmanager

@contextmanager
def temp_setattr(obj, attr, val):
    val_ = getattr(obj, attr)
    try:
        setattr(obj, attr, val)
        yield
    finally:
        setattr(obj, attr, val_)

def strong_convex_func(x, lamb, useHingedL2=False):
    if useHingedL2:
        func = (torch.maximum(x, 0) ** 2) / lamb / 2.
    else:
        func = torch.exp(x / lamb) / math.exp(1) * lamb
    return func

def strong_convex_func_normalized(x, lamb, useHingedL2=False):
    if useHingedL2:
        func = (torch.maximum(x, 0) ** 2) / 2.
    else:
        func = torch.exp(x / lamb) / math.exp(1)
    return func

def sum_probs_func(x, lamb):
    return torch.mean(torch.maximum(x, 0.0)) / lamb

def first_element(input):
    """Improve compatibility of single and multiple output components.
    """
    if type(input) == tuple or type(input) == list:
        return input[0]
    else:
        return input

def cond_cosine(x1, x2):
    def cosine_cost(x, y):
        cos = nn.CosineSimilarity()
        return 1 - cos(x.unsqueeze(-1), y.unsqueeze(-1).transpose(0, -1))
    return cosine_cost(x1[0].flatten(1), x2[0].flatten(1)) \
        + torch.cdist(x1[1], x2[1], p=1) / x1[1].shape[-1]