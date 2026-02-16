import torch
from collections.abc import Sequence, Callable, Iterable
from typing import Optional
import math
class AdamW(torch.optim.Optimizer):
    def __init__(self, params: torch.nn.Parameter,lr:float, weight_decay:float, betas: Sequence[float], eps: float):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params=params,defaults=defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["t"] = 1
                m = state["m"]
                v = state["v"]
                t = state["t"]
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad**2)
                lr_t = lr * math.sqrt(1 - math.pow(beta2, t))/(1 - math.pow(beta1, t))
                p.data -= lr_t * m / (v.sqrt() + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v 
        return loss