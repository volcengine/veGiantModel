# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
import sys
import os

cwd = os.path.dirname(os.path.abspath(__file__))
_deepspeed_dir = os.path.join(cwd, '../../third_party/deepspeed')
_megatron_dir = os.path.join(cwd, '../../third_party/megatron')
sys.path.append(cwd)
sys.path.append(_deepspeed_dir)
sys.path.append(_megatron_dir)

from . import patcher
from .engine.engine import VeGiantModelEngine
from .initialize import initialize_megatron, init_distribute
from .distributed import *

def initialize(args,
               model,
               optimizer=None,
               model_parameters=None,
               training_data=None,
               lr_scheduler=None,
               mpu=None,
               dist_init_required=None,
               collate_fn=None,
               config_params=None):
    engine = VeGiantModelEngine(args=args,
                    model=model,
                    optimizer=optimizer,
                    model_parameters=model_parameters,
                    training_data=training_data,
                    lr_scheduler=lr_scheduler,
                    mpu=model.mpu(),
                    dist_init_required=dist_init_required,
                    collate_fn=collate_fn,
                    config_params=config_params)

    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler
    ]
    return tuple(return_items)
