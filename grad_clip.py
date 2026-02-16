import torch
import torch.nn as nn
from collections.abc import Iterable


def clip_grad(params: Iterable[nn.Parameter], max_l2_norm: float):
    total_norm = 0.0
    for param in params:
        if param.grad is not None:
            total_norm += param.grad.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / total_norm
        for param in params:
            if param.grad is not None:
                param.grad.mul_(clip_coef)
