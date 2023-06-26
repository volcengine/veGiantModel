# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
import sys
import torch

from veturbollm import dist_signal_handler
from veturbollm.tokenizer import build_tokenizer
from veturbollm.microbatches import build_num_microbatches_calculator
from veturbollm.utils.timers import Timers
from veturbollm.config import TaskConfig

_GLOBAL_ARGS = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_TIMERS = None
_GLOBAL_SIGNAL_HANDLER = None


def get_args() -> TaskConfig:
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, "args")
    return _GLOBAL_ARGS


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples, consistency_check)


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, "tokenizer")
    return _GLOBAL_TOKENIZER


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, "timers")
    return _GLOBAL_TIMERS


def get_signal_handler():
    _ensure_var_is_initialized(_GLOBAL_SIGNAL_HANDLER, "signal handler")
    return _GLOBAL_SIGNAL_HANDLER


def _set_signal_handler():
    global _GLOBAL_SIGNAL_HANDLER
    _ensure_var_is_not_initialized(_GLOBAL_SIGNAL_HANDLER, "signal handler")
    _GLOBAL_SIGNAL_HANDLER = dist_signal_handler.DistributedSignalHandler().__enter__()


def set_global_variables(args):
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""

    assert args is not None

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, "args")
    set_args(args)

    _build_num_microbatches_calculator(args)
    # if args.vocab_file:
    _ = _build_tokenizer(args)
    _set_tensorboard_writer(args)
    _set_timers(args)

    # if args.exit_signal_handler:
    #     _set_signal_handler()


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def _build_num_microbatches_calculator(args):
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _ensure_var_is_not_initialized(_GLOBAL_NUM_MICROBATCHES_CALCULATOR, "num microbatches calculator")

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(args)


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, "tokenizer")
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER, "tensorboard writer")

    if args.training.tensorboard_dir and args.rank == (args.world_size - 1):
        try:
            if os.environ.get("MLP_TRACKING_ENABLE", "false") in ["true", "True", "1"]:
                import volcengine_ml_platform
                from volcengine_ml_platform import wandb

                print("> setting tracking ...")
                volcengine_ml_platform.init()
                project_name = os.environ.get("MLP_TRACKING_PROJECT_NAME", "veturbollm")
                wandb.init(project=project_name, sync_tensorboard=True)
                args = get_args()
                wandb.config.update(get_args())
            from torch.utils.tensorboard import SummaryWriter

            print("> setting tensorboard ...")
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.training.tensorboard_dir, max_queue=args.training.tensorboard_queue_size
            )
        except ModuleNotFoundError:
            print(
                "WARNING: TensorBoard writing requested but is not "
                "available (are you using PyTorch 1.1.0 or later?), "
                "no TensorBoard logs will be written.",
                flush=True,
            )


def _set_timers(args):
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, "timers")
    _GLOBAL_TIMERS = Timers(args.logging.timing_log_level, args.logging.timing_log_option)


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, "{} is already initialized.".format(name)
