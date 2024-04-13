
import torch

from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
from src.model.util import clip_gradient, get_cosine_annealing_step_size

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}   
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, weight_decay, betas, eps, max_grad_norm, alpha_min, alpha_max, T_warmup, T_cosine):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.T_warmup = T_warmup
        self.T_cosine = T_cosine
        defaults = {"betas": betas, "weight_decay": weight_decay, "eps": eps, "max_grad_norm": max_grad_norm}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                # self.state[p] = {"m": torch.zeros_like(p.data),
                #                     "v": torch.zeros_like(p.data)}
                self.state[p]['step'] = 0
                self.state[p]['m'] = torch.zeros_like(p.data)
                self.state[p]['v'] = torch.zeros_like(p.data)


    # def step(self, closure: Optional[Callable] = None):
    #     loss = None if closure is None else closure()
    #     for group in self.param_groups:
    #         lr = group["lr"] # Get the learning rate.
    #         for p in group["params"]:
    #             if p.grad is None:
    #                 continue
    #             grad = p.grad.data
    #             state = self.state[p] # Get state associated with p.
    #             state["m"] = (self.betas[0] * state["m"]) + (1 - self.betas[0]) * grad
    #             state["v"] = (self.betas[1] * state["v"]) + (1 - self.betas[1]) * grad**2
    #             t = state.get("t", 0)
    #             alpha_t = lr * (1 - self.betas[1]**(t + 1)) ** 0.5 / (1 - self.betas[0]**(t + 1))
    #             grad = p.grad.data # Get the gradient of loss with respect to p.
    #             p.data -= alpha_t / math.sqrt(t + 1) * state["m"] / (torch.sqrt(state["v"]) + self.eps) # Update weight tensor in-place.
    #             p.data -= lr * self.weight_decay * p.data
    #             state["t"] = t + 1 # Increment iteration number.
    #     return loss
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            clip_gradient([p.grad for p in group['params']], 1.0)
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                beta1, beta2 = group['betas']

                state['step'] += 1
                state["m"] = (beta1 * state["m"]) + (1 - beta1) * grad
                state["v"] = (beta2 * state["v"]) + (1 - beta2) * grad**2

                lr = get_cosine_annealing_step_size(state["step"], alpha_min=self.alpha_min, alpha_max=self.alpha_max, T_w=self.T_warmup, T_c=self.T_cosine)

                step_size = lr * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])

                p.data -= step_size * state["m"] / (torch.sqrt(state["v"]) + group["eps"]) # Update weight tensor in-place.
                p.data -= lr * group["weight_decay"] * p.data

        return loss
