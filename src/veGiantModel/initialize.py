# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
import torch
import os
import random
import numpy as np

from megatron.global_vars import set_global_variables
from megatron import get_args, mpu, print_rank_0
from .engine.topology import PipeModelDataParallelTopology, PipelineParallelGrid
from .launcher.launch import launch_bps
from deepspeed.utils import log_dist
import logging

_GLOBAL_EXTRA_ARGS = None

def add_byte_giant_model_customize_args(parser):
    import deepspeed
    parser = deepspeed.add_config_arguments(parser)
    group = parser.add_argument_group(title='bytedance')
    group.add_argument('--cpu-optimizer', action='store_true',
                       help='Run optimizer on CPU')
    group.add_argument('--cpu_torch_adam', action='store_true',
                       help='Use Torch Adam as optimizer on CPU.')
    group.add_argument('--vocab-size', type=int, default=1000,
                       help='vocab size.')
    group.add_argument('--train-batch-size', type=int, default=0,
                       help='global batch size')
    group.add_argument('--train_micro_batch_size_per_gpu', type=int, default=0,
                       help='Batch size per model instance (for deepspeed). '
                       'Global batch size is local batch size times data '
                       'parallel size.')
    group.add_argument('--deepspeed-activation-checkpointing', action='store_true',
                       help='deepspeed_activation_checkpointing.')
    group.add_argument('--deepspeed-pipeline', action='store_true',
                       help='enable pipeline parallelism via deepspeed.')
    group.add_argument('--ci', action='store_true', help="run in CI environment")
    group.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="set gradient_accumulation_steps for deepspeed config")
    group.add_argument('--train_batch_size', type=int, default=0,
                        help="train_batch_size")
    group.add_argument('--broadcast_activation', action='store_true', help="use broadcast to send/recv activation")
    group.add_argument('--broadcast_grads', action='store_true', help="use broadcast to send/recv grads")
    group.add_argument('--partition_method', type=str, default='uniform',
                       help='the method to partition layers in pipeline parallelism.')
    group.add_argument('--config_param', type=str, default='',
                       help='json dict for deepspeed config')

    group.add_argument('--num-stages', type=int, default=1,
                       help='number of stages')

    group.add_argument('--load-megatron', type=str, default=None,
                       help='Directory containing a model checkpoint in Megatron format.')
    
    if _GLOBAL_EXTRA_ARGS is not None:
        parser = _GLOBAL_EXTRA_ARGS(parser)

    return parser

def initialize_megatron(extra_args_provider=None, args_defaults={}):
    if extra_args_provider is not None:
        global _GLOBAL_EXTRA_ARGS
        _GLOBAL_EXTRA_ARGS = extra_args_provider

    set_global_variables(extra_args_provider=add_byte_giant_model_customize_args, args_defaults=args_defaults)
    args = get_args()
    init_distribute(args.num_stages, args.model_parallel_size)
    _set_random_seed(args.seed)

def _init_topology(num_stages, mp_size):
    num_pp = num_stages
    num_mp = mp_size
    num_dp = (torch.distributed.get_world_size() // num_pp) // num_mp
    log_dist('rank: {args.rank}, init topology with num_pp:{num_pp}, num_mp:{num_mp}, \
        num_dp: {num_dp}', ranks=[-1], level=logging.DEBUG)
    topology = PipeModelDataParallelTopology(num_pp=num_pp, num_mp=num_mp, num_dp=num_dp)
    log_dist(f'finish building topology, topology.mapping: {topology.mapping}', \
        ranks=[-1], level=logging.DEBUG)
    return PipelineParallelGrid(topology)

def _set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            mpu.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed))

def init_distribute(num_stages, mp_size,
                    distributed_backend='nccl', init_method='tcp://'):
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))
    device_count = torch.cuda.device_count()
    local_rank = rank % device_count

    if torch.distributed.is_initialized():
        print_rank_0('torch distributed is already initialized, '
                'skipping initialization ...')
    else:
        print_rank_0('> initializing torch distributed ...')
       
        torch.cuda.set_device(local_rank)
        # Call the init process
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method += master_ip + ':' + master_port
        torch.distributed.init_process_group(
            backend=distributed_backend,
            world_size=world_size, rank=rank,
            init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    grid = _init_topology(num_stages, mp_size)
    mpu.initialize_model_parallel(grid)
    if num_stages > 1:
        import byteps.torch as bps
        assert bps is not None
        launch_bps(local_rank)
