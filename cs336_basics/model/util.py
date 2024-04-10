import torch
import math
import numpy as np
import os
import typing

def softmax(x, dim=-1):
    x -= x.max(dim=-1, keepdim=True)[0] # for numerical stability
    x = torch.exp(x)
    return x / x.sum(dim=-1, keepdim=True)

def crossEntropyLossUnstable(y, y_hat_logits): # y of shape (batch_size) y_hat_logits of shape (batch_size, vocab_size)
    y_hat = softmax(y_hat_logits, dim=-1)
    true_class_probs = torch.gather(y_hat, 1, y.unsqueeze(1)).squeeze(1)
    neg_log_probs = - torch.log(true_class_probs)
    return torch.mean(neg_log_probs)

def crossEntropyLoss(y, y_hat_logits): # y of shape (batch_size) y_hat_logits of shape (batch_size, vocab_size)
    y_hat_logits -= y_hat_logits.max(dim=-1, keepdim=True)[0] # for numerical stability
    log_sum_exp = torch.log(torch.sum(torch.exp(y_hat_logits), dim=1))
    true_class_logits = torch.gather(y_hat_logits, 1, y.unsqueeze(1)).squeeze(1)

    loss = log_sum_exp - true_class_logits
    return torch.mean(loss)

def get_cosine_annealing_step_size(curr_iter, alpha_min, alpha_max, T_w, T_c):
    if curr_iter < T_w:
        return alpha_max * curr_iter / T_w
    if curr_iter > T_c:
        return alpha_min
    return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos((curr_iter - T_w) / (T_c - T_w) * math.pi))

def clip_gradient(parameters_list, max_l2_norm):
    for parameters in parameters_list:
        norm = torch.norm(parameters.data, p=2)
        if norm > max_l2_norm:
            parameters.data = parameters.data * max_l2_norm / (norm + 1e-6)

def get_batch(data, batch_size, context_length, device):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system.")
        device_index = int(device.split(":")[-1])  # Extract device index
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device with index {device_index} does not exist.")
        
    def rand_sample():
        index = np.random.randint(0, len(data) - context_length)
        return [data[index: index + context_length], data[index + 1: index + context_length + 1]]
    
    samples = np.array([rand_sample() for _ in range(batch_size)]) # (batch_size, 2, context_length)

    try:
        torch_samples = torch.tensor(samples, device=device).transpose(0,1) # (2, batch_size, context_length)
    except RuntimeError as e:
        print(f"Failed to create tensor on device {device}: {e}")
        raise
    return torch_samples[0], torch_samples[1]


def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
                    ):
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }

    torch.save(checkpoint, out)

def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer
        ):
    
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration