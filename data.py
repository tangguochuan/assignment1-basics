import numpy.typing as npt
import torch
import numpy as np
def get_batch(dataset: npt.NDArray, batch_size: int, seq_len: int, device:str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start_idx = len(dataset) - seq_len - 1
    start_indices = np.random.randint(0, max_start_idx + 1, size = batch_size)
    inputs_list = []
    target_lists = []

    for start_idx in start_indices:
        inputs_list.append(dataset[start_idx : start_idx + seq_len])
        target_lists.append(dataset[start_idx + 1 : start_idx + seq_len + 1])
    inputs = torch.tensor(np.array(inputs_list), device=device)
    targets = torch.tensor(np.array(target_lists), device=device)
    return inputs, targets