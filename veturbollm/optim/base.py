import torch
import torch.nn as nn
from transformers import get_scheduler

from veturbollm.global_vars import get_args


def get_optimizer_with_scheduler(model: nn.Module):
    args = get_args()

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.regularization.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate.lr)

    lr_scheduler = get_scheduler(
        name=args.learning_rate.lr_decay_style,
        optimizer=optimizer,
        num_warmup_steps=args.learning_rate.lr_warmup_iters * args.training.gradient_accumulation_steps,
        num_training_steps=args.training.train_iters * args.training.gradient_accumulation_steps,
    )
    return optimizer, lr_scheduler
