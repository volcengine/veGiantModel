# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

import os
import torch
import torch.distributed as dist
from deepspeed.utils import logger, log_dist

ENABLE_PYTORCH_BROADCAST = os.environ.get("ENABLE_PYTORCH_BROADCAST", "0") != "0"

try:
    if not ENABLE_PYTORCH_BROADCAST:
        import byteps.torch as bps
    else:
        print("BytePS import is disabled", flush=True)
        bps = None
except ImportError:
    print("BytePS is not installed")
    bps = None

_groups = None
_grid = None

DS_PIPE_VERBOSE = os.environ.get('DS_PIPE_VERBOSE', "0") != "0"

did_recv = False
send_stream = None
recv_stream = None 


bps_send_handles = {}
bps_recv_handles = {}


#initializes adjacent process groups
#run this only after torch.distributed.init_process_group() has been called
def init_process_groups(grid):
    global _groups, _grid
    _grid = grid

    assert _grid.pipe_parallel_size > 1, "There is no model parallelism"

    _groups = [dist.new_group(ranks=group) for group in _grid.p2p_groups]


def _is_valid_send_recv(src_stage, dest_stage):
    first_stage = 0
    last_stage = _grid.pipe_parallel_size - 1
    assert abs(src_stage-dest_stage) == 1 or \
        (src_stage == first_stage and dest_stage == last_stage) or \
        (src_stage == last_stage and dest_stage == first_stage), \
    "Functionality currently limited to send and receive between adjacent ranks only"


def send(tensor, dest_stage, async_op=False):
    global _groups

    async_op = False
    src_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    group = _get_send_recv_group(src_stage, dest_stage)
    src_rank = _grid.stage_to_global(stage_id=src_stage)

    import torch
    if tensor.dtype != torch.float32 and DS_PIPE_VERBOSE:
        print('warning: p2p send', tensor.dtype, tensor.shape, flush=True)
    return _send(tensor, src_rank, group, async_op)

def _bps_get_name(src, dest, name, suffix):
    return "_".join([str(src), str(dest), str(name), str(suffix)])

def bps_send(tensor, dest_stage, name, index, async_op=True):
    global bps_send_handles

    src_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)
    src_rank = _grid.stage_to_global(stage_id=src_stage)
    dest_rank = _grid.stage_to_global(stage_id=dest_stage)
    name = _bps_get_name(src_rank, dest_rank, name, index)
    if name not in bps_send_handles:
        # XXX hard-code max number of tensors for this name
        bps_send_handles[name] = [None] * 10
    else:
        handle = bps_send_handles[name][index]
        if handle is not None:
            bps.synchronize(handle)
    handle = bps.send_async(tensor, dest_rank, name=name)
    # XXX
    if not async_op:
        bps.synchronize(handle)
    else:
        bps_send_handles[name][index] = handle
    return tensor

def bps_sync(src_stage, name, index=0):
    dest_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)
    src_rank = _grid.stage_to_global(stage_id=src_stage)
    dest_rank = _grid.stage_to_global(stage_id=dest_stage)
    name = _bps_get_name(src_rank, dest_rank, name, index)
    if name in bps_recv_handles:
        handle = bps_recv_handles[name][index]
        if handle is not None:
            bps.synchronize(handle)

def bps_sync_all():
    for name, handles in bps_send_handles.items():
        for handle in handles:
            if handle is not None:
                bps.synchronize(handle)

    for name, handles in bps_recv_handles.items():
        for handle in handles:
            if handle is not None:
                bps.synchronize(handle)

def bps_recv(tensor, src_stage, name, index=0, async_op=True):
    global bps_recv_handles

    dest_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)
    src_rank = _grid.stage_to_global(stage_id=src_stage)
    dest_rank = _grid.stage_to_global(stage_id=dest_stage)
    name = _bps_get_name(src_rank, dest_rank, name, index)
    if name not in bps_recv_handles:
        # XXX hard-code max number of tensors for this name
        bps_recv_handles[name] = [None] * 10
    else:
        handle = bps_recv_handles[name][index]
        if handle is not None:
            bps.synchronize(handle)
    handle = bps.recv_async(tensor, src_rank, name=name)
    if not async_op:
        bps.synchronize(handle)
    else:
        bps_recv_handles[name][index] = handle
    return tensor


def _send(tensor, src_rank, group, async_op):
    global did_recv
    return dist.broadcast(tensor, src_rank, group=group, async_op=async_op)

def send_grads(tensor, grid, async_op=False):
    async_op = False
    if  grid.send_grads_src_rank == grid.global_rank:
        # print(f'start rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _send_grad_src_rank: {grid.send_grads_src_rank}, send group: {grid.send_grads_group}, send_grad_groups: {grid.send_grads_proc_group}', flush=True)
        _send(tensor, grid.send_grads_src_rank, grid.send_grads_proc_group, async_op)
        # print(f'finis rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _send_grad_src_rank: {grid.send_grads_src_rank}, send group: {grid.send_grads_group}', flush=True)
    else:
        # print(f'finish fast rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _send_grad_src_rank: {grid.send_grads_src_rank}, send group: {grid.send_grads_group}', flush=True)
        pass

def _recv(tensor, src_rank, group, async_op):
    global did_recv
    tensor = dist.broadcast(tensor, src_rank, group=group, async_op=async_op)
    did_recv = True
    return tensor

def recv_grads(tensor, grid, async_op=False):
    async_op = False
    # print(f'start rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _recv_grad_src_rank: {grid.recv_grads_src_rank}, recv group: {grid.recv_grads_group}, recv_grad_groups: {grid.recv_grads_proc_group}', flush=True)
    _recv(tensor, grid.recv_grads_src_rank, grid.recv_grads_proc_group, async_op)
    # print(f'finish rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _recv_grad_src_rank: {grid.recv_grads_src_rank}, recv group: {grid.recv_grads_group}', flush=True)


def send_activations(tensor, grid, async_op=False):
    async_op = False
    if  grid.send_activation_src_rank == grid.global_rank:
        # print(f'start rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _send_grad_src_rank: {grid.send_grads_src_rank}, send group: {grid.send_grads_group}, send_grad_groups: {grid.send_grads_proc_group}', flush=True)
        _send(tensor, grid.send_activation_src_rank, grid.send_activation_proc_group, async_op)
        # print(f'finis rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _send_grad_src_rank: {grid.send_grads_src_rank}, send group: {grid.send_grads_group}', flush=True)
    else:
        # print(f'finish fast rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _send_grad_src_rank: {grid.send_grads_src_rank}, send group: {grid.send_grads_group}', flush=True)
        pass 

def recv_activations(tensor, grid, async_op=False):
    async_op = False
    # print(f'start rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _recv_grad_src_rank: {grid.recv_grads_src_rank}, recv group: {grid.recv_grads_group}, recv_grad_groups: {grid.recv_grads_proc_group}', flush=True)
    _recv(tensor, grid.recv_activation_src_rank, grid.recv_activation_proc_group, async_op)
    # print(f'finish rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _recv_grad_src_rank: {grid.recv_grads_src_rank}, recv group: {grid.recv_grads_group}', flush=True)

def recv(tensor, src_stage, async_op=False):
    global _groups
    global did_recv

    async_op = False
    dest_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    group = _get_send_recv_group(src_stage, dest_stage)
    src_rank = _grid.stage_to_global(stage_id=src_stage)
    return _recv(tensor, src_rank, group, async_op)


def barrier(stage_id):
    global _groups, _grid
    group_id = _grid.stage_to_global(stage_id=stage_id)
    if (dist.get_rank() >= 0):
        print("Barrier Group ID", group_id)
        print("Barrier Group", _grid.p2p_groups[group_id])
    dist.barrier(group=_groups[group_id])
    if (dist.get_rank() >= 0):
        print("Exiting Barrier ", group_id)


def _get_send_recv_group(src_stage, dest_stage):
    '''the group id is always the smaller rank unless its a wrap around'''

    stage_id = None

    first_stage = 0
    last_stage = _grid.pipe_parallel_size - 1

    if (src_stage == first_stage and dest_stage == last_stage
            or dest_stage == first_stage and src_stage == last_stage):
        stage_id = last_stage
    elif src_stage > dest_stage:
        stage_id = dest_stage
    else:
        stage_id = src_stage
    '''group_id corresponds to group of [group_id, group_id+1]
     unless group_id is the rank of the last stage
     in which case group_id correspods to group[group_id-num_stages+1, group_id]
     '''
    group_id = _grid.stage_to_global(stage_id=stage_id)

    return _groups[group_id]
