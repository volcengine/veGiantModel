import os
from datetime import timedelta

import torch
from omegaconf import OmegaConf

from veturbollm.global_vars import set_global_variables
from veturbollm.config import TaskConfig


def load_config(config_file):
    user_conf = OmegaConf.load(config_file)

    # Load a default config from current directory
    # default_conf = OmegaConf.load(os.path.join(os.path.dirname(__file__), 'misc', 'default_conf.yaml'))
    user_conf = TaskConfig(**user_conf)

    return user_conf


def initialize_distributed(args):
    # Args from environment
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = args.rank % torch.cuda.device_count()

    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6000")
    init_method += master_ip + ":" + master_port

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if args.rank == 0:
            print(
                "torch distributed is already initialized, skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:
        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device
            torch.cuda.set_device(device)

    # Call the init process
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=args.world_size,
        rank=args.rank,
        init_method=init_method,
        timeout=timedelta(minutes=30),
    )


def initialize_veturbollm(config_file):
    args = load_config(config_file)

    initialize_distributed(args)

    # Set the tensor model parallel size
    model_parallel_size = args.distributed.pipeline_model_parallel_size * args.distributed.tensor_model_parallel_size
    assert args.world_size % model_parallel_size == 0, (
        "world size is not"
        " divisible by tensor parallel size ({}) times pipeline parallel "
        "size ({})".format(
            args.world_size,
            args.distributed.tensor_model_parallel_size,
            args.distributed.pipeline_model_parallel_size,
        )
    )
    args.distributed.data_parallel_size = args.world_size // model_parallel_size

    # conf = OmegaConf.merge(default_conf, user_conf)
    if args.training.global_batch_size is not None:
        args.training.gradient_accumulation_steps = args.training.global_batch_size // (
            args.training.micro_batch_size * args.distributed.data_parallel_size
        )
        print(f"reset gradient_accumulation_steps to {args.training.gradient_accumulation_steps}")
    else:
        args.training.global_batch_size = (
            args.training.micro_batch_size
            * args.distributed.data_parallel_size
            * args.training.gradient_accumulation_steps
        )

    set_global_variables(args)
    return
