import math
def lr_cosine_schedule(it: int, max_lr: float, min_lr: float, warm_iters: int, cosine_iters: int):
    if it < warm_iters:
        return it * max_lr / warm_iters
    
    if it >= warm_iters and it <= cosine_iters:
        return min_lr + 0.5 * (1 + math.cos((it - warm_iters)* math.pi / (cosine_iters - warm_iters))) * (max_lr - min_lr)
    
    return min_lr