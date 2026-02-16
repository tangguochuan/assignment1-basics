import torch
def CrossEntropyLoss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    inputs: [... vocab_size]
    targets: [...]
    """
    log_sum_exp = torch.logsumexp(inputs, dim=-1)
    target_logits = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (-target_logits + log_sum_exp).mean()
    