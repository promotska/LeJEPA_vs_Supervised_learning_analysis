from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> LambdaLR:
    warmup_epochs = int(max(0, warmup_epochs))
    total_epochs = int(max(1, total_epochs))

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)