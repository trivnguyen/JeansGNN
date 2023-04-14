
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class AttentionScheduler(_LRScheduler):
    """ Learning rate scheduler for Attention-MAF """

    def __init__(
            self, optimizer: Optimizer, dim_embed: int,
            warmup_steps: int, last_epoch: int = -1,
            verbose: bool = False
        ) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super(AttentionScheduler, self).__init__(
            optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = self._calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups

    def _calc_lr(self, step, dim_embed, warmup_steps) -> float:
        return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))
