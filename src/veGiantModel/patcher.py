# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import torch
print("Loading veGiantModel submodules ...")

_TOPOLOGY = None

def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _TOPOLOGY is None


def initialize_model_parallel(grid):
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    global _TOPOLOGY
    _TOPOLOGY = grid


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TOPOLOGY is None:
        return False
    return True


def get_model_parallel_group():
    """Get the parallel group the caller rank belongs to."""
    assert _TOPOLOGY is not None, \
        ' parallel group is not initialized'
    return _TOPOLOGY.get_slice_parallel_group()


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _TOPOLOGY is not None, \
        'data parallel group is not initialized'
    return _TOPOLOGY.get_data_parallel_group()


def set_model_parallel_world_size(world_size):
    pass


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    return _TOPOLOGY.get_slice_parallel_world_size()


def set_model_parallel_rank(rank):
    pass


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    return _TOPOLOGY.get_slice_parallel_rank()


def get_model_parallel_src_rank():
    return _TOPOLOGY.get_slice_parallel_src_rank()


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return _TOPOLOGY.get_data_parallel_world_size()


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return _TOPOLOGY.get_data_parallel_rank()

def get_pipe_parallel_rank():
    return _TOPOLOGY.get_pipe_parallel_rank()

def destroy_model_parallel():
    """Set the groups to none."""
    global _TOPOLOGY
    _TOPOLOGY = None

def get_grid():
    return _TOPOLOGY

def get_topo():
    return _TOPOLOGY.topology()

import megatron.mpu.initialize as initialize
initialize.is_unitialized = is_unitialized
initialize.initialize_model_parallel = initialize_model_parallel
initialize.model_parallel_is_initialized = model_parallel_is_initialized
initialize.get_model_parallel_group = get_model_parallel_group
initialize.get_data_parallel_group = get_data_parallel_group
initialize.set_model_parallel_world_size = set_model_parallel_world_size
initialize.get_model_parallel_world_size = get_model_parallel_world_size
initialize.set_model_parallel_rank = set_model_parallel_rank
initialize.get_model_parallel_rank = get_model_parallel_rank
initialize.get_model_parallel_src_rank = get_model_parallel_src_rank
initialize.get_data_parallel_world_size = get_data_parallel_world_size
initialize.get_data_parallel_rank = get_data_parallel_rank
initialize.get_pipe_parallel_rank = get_pipe_parallel_rank
initialize.destroy_model_parallel = destroy_model_parallel

from megatron import mpu
from importlib import reload  
reload(mpu.data)
reload(mpu.mappings)
reload(mpu.cross_entropy)
mpu.get_pipe_parallel_rank = get_pipe_parallel_rank
reload(mpu)

from megatron.mpu import mappings

def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    group = get_model_parallel_group()
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output

mappings._gather = _gather

from megatron.tokenizer import tokenizer as token
from megatron.tokenizer.tokenizer import _BertWordPieceTokenizer, _vocab_size_with_padding, _GPT2BPETokenizer

def build_tokenizer(args):
    if args.vocab_file is None:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size,
                                                    args)
        return None
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type),
              flush=True)

    # Select and instantiate the tokenizer.
    assert args.vocab_file is not None
    if args.tokenizer_type == 'BertWordPieceLowerCase':
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=True)
    elif args.tokenizer_type == 'BertWordPieceCase':
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=False)
    elif args.tokenizer_type == 'GPT2BPETokenizer':
        assert args.merge_file is not None
        tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(args.tokenizer_type))

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size,
                                                      args)

    return tokenizer

token.build_tokenizer = build_tokenizer
import megatron
reload(megatron.tokenizer)
reload(megatron.global_vars)
reload(megatron.global_vars)
print("veGiantModel loaded.")
