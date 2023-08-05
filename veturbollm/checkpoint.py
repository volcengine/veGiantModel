import json
import os

import torch
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig
from veturbollm.utils.tools import print_rank_0
from veturbollm.config import TaskConfig


def save_checkpoint(model, optimizer, lr_scheduler, args: TaskConfig):
    save_dir = os.path.join(args.checkpointing.save, f"step-{args.completed_steps}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    unwrap_model = model.model

    if args.distributed.strategy == "fsdp":
        FSDP.set_state_dict_type(
            unwrap_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        )
        state_dict = unwrap_model.state_dict()
        optim_state_dict = FSDP.optim_state_dict(unwrap_model, optimizer)
    else:
        state_dict = unwrap_model.state_dict()
        optim_state_dict = optimizer.state_dict()

    if args.rank != 0:
        return

    # TODO: rewrite save_pretrained for whole pipeline
    unwrap_model.config.save_pretrained(save_dir)

    save_path = os.path.join(save_dir, "pytorch_model.bin")
    torch.save(state_dict, save_path)

    save_path = os.path.join(save_dir, "optim.bin")
    torch.save(optim_state_dict, save_path)

    save_path = os.path.join(save_dir, "lr_scheduler.bin")
    torch.save(lr_scheduler.state_dict(), save_path)

    save_path = os.path.join(save_dir, "args.json")
    with open(save_path, "w") as f:
        json.dump(args.dict(), f)

    with open(os.path.join(args.checkpointing.save, "completed_steps.txt"), "w") as f:
        f.write(str(args.completed_steps))
    print_rank_0(f"Saved checkpoint to {args.checkpointing.save}")
    return save_path


def resume_status(args: TaskConfig):
    if not os.path.exists(os.path.join(args.checkpointing.load, "completed_steps.txt")):
        print_rank_0(f"Checkpoint not found at {args.checkpointing.load}")
        return

    with open(os.path.join(args.checkpointing.load, "completed_steps.txt"), "r") as f:
        completed_steps = int(f.read())
    print_rank_0(f"Resuming from step {completed_steps}")

    args.completed_steps = completed_steps
    args.consumed_train_samples = completed_steps * args.training.global_batch_size
    args.model.pretrained_model_name_or_path = os.path.join(args.checkpointing.load, f"step-{completed_steps}")


def resume_optimizer(model, optimizer, lr_scheduler, args: TaskConfig):
    optim_path = os.path.join(args.model.pretrained_model_name_or_path, "optim.bin")
    if not os.path.exists(optim_path):
        print_rank_0(f"Optimizer checkpoint not found at {optim_path}")
        return

    unwrap_model = model.model

    if args.distributed.strategy == "fsdp":
        if args.rank == 0:
            optim_state_dict = torch.load(optim_path)
        else:
            # we will use FSDP to sync optimizer state dict
            optim_state_dict = {}
        # TODO: there are many bugs/typos in pytorch 2.0, and will be fixed on pytorch 2.1
        # FSDP.set_state_dict_type(
        #     unwrap_model,
        #     StateDictType.FULL_STATE_DICT,
        #     FullStateDictConfig(rank0_only=True),
        #     FullOptimStateDictConfig(rank0_only=True),
        # )
        # optim_state_dict = FSDP.optim_state_dict_to_load(optim_state_dict, unwrap_model, optimizer)

        optim_state_dict = FSDP._optim_state_dict_to_load_impl(
            optim_state_dict=optim_state_dict,
            model=unwrap_model,
            optim_input=None,
            optim=optimizer,
            full_state_dict=True,
            rank0_only=True,
        )
    else:
        # for ddp
        optim_state_dict = torch.load(optim_path)
    optimizer.load_state_dict(optim_state_dict)

    lr_scheduler_path = os.path.join(args.model.pretrained_model_name_or_path, "lr_scheduler.bin")
    if os.path.exists(lr_scheduler_path):
        lr_scheduler.load_state_dict(torch.load(lr_scheduler_path))
