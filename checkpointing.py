import torch
import torch.nn as nn
import os
import typing
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iterations: int, out: str| os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iterations": iterations
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iterations"]