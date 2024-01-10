
import torch
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

class AttentionScheduler(LambdaLR):
    """ Learning rate scheduler for Attention-MAF """

    def __init__(
            self, optimizer: Optimizer, dim_embed: int,
            warmup_steps: int, last_epoch: int = -1,
            verbose: bool = False
        ) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(
            optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = self._calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups

    def _calc_lr(self, step, dim_embed, warmup_steps) -> float:
        return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))


class WarmUpCosineAnnealingLR(LambdaLR):
    def __init__(self, optimizer, decay_steps, warmup_steps, eta_min=0, last_epoch=-1):
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.eta_min + (
            0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps))))

