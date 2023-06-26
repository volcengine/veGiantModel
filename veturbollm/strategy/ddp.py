import torch.nn as nn
import torch
from .base import Strategy


class DDPStrategy(Strategy):
    def setup_model_and_optimizer(self, model, optimizer=None):
        local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(local_rank),
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

        return model, optimizer
