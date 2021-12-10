from .. import patcher as dist
from megatron import mpu

def get_model_parallel_world_size():
    return dist.get_model_parallel_world_size()

def get_model_parallel_rank():
    return dist.get_model_parallel_rank()

def get_data_parallel_world_size():
    return dist.get_data_parallel_world_size()

def get_model_parallel_group():
    return dist.get_model_parallel_group()

def get_grid():
    return dist.get_grid()

def copy_to_model_parallel_region(input_):
    return mpu.copy_to_model_parallel_region(input_)

def reduce_from_model_parallel_region(input_):
    return mpu.reduce_from_model_parallel_region(input_)

def gather_from_model_parallel_region(input_):
    return mpu.gather_from_model_parallel_region(input_)
