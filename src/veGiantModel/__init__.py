# Copyright (c) 2021, ByteDance Inc.  All rights reserved.

from .engine.engine import veGiantModelEngine


def add_ve_giant_model_customize_args(parser):
    group = parser.add_argument_group(title='veGiantModel')
    group.add_argument('--vegiantmodel', action='store_true', help='Enable veGiantModel')
    group.add_argument('--broadcast-activation', action='store_true', help="use broadcast to send/recv activation")
    group.add_argument('--broadcast-grads', action='store_true', help="use broadcast to send/recv grads")
    # group.add_argument('--partition_method', type=str, default='uniform',
    #                    help='the method to partition layers in pipeline parallelism.')
    # group.add_argument('--config_param', type=str, default='', help='json dict for deepspeed config')
    # group.add_argument('--num-stages', type=int, default=1, help='number of stages')
    return parser


def initialize(args,
               model,
               optimizer=None,
               model_parameters=None,
               training_data=None,
               lr_scheduler=None,
               dist_init_required=None,
               collate_fn=None,
               config_params=None):
    engine = veGiantModelEngine(args=args,
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
