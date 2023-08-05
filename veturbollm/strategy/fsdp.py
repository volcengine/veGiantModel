import functools

import torch
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, BackwardPrefetch

from veturbollm.utils.dtype import get_torch_dtype

from .base import Strategy


def get_mixed_precision(precision, mixed_precision="DEFAULT", keep_low_precision_grads=False):
    """Helper function for configuring mixed_precision."""
    param_dtype = None
    reduce_dtype = None
    buffer_dtype = None
    if isinstance(mixed_precision, dict):
        param_dtype = mixed_precision.get("param_dtype", None)
        if param_dtype is not None:
            param_dtype = get_torch_dtype(param_dtype)
        reduce_dtype = mixed_precision.get("reduce_dtype", None)
        if reduce_dtype is not None:
            reduce_dtype = get_torch_dtype(reduce_dtype)
        buffer_dtype = mixed_precision.get("buffer_dtype", None)
        if buffer_dtype is not None:
            buffer_dtype = get_torch_dtype(buffer_dtype)
    elif isinstance(mixed_precision, str):
        mixed_precision = mixed_precision.upper()
        if mixed_precision == "FULL":
            pass
        elif mixed_precision == "DEFAULT":
            param_dtype = get_torch_dtype(precision)
            buffer_dtype = get_torch_dtype(precision)
        elif mixed_precision == "PURE":
            param_dtype = get_torch_dtype(precision)
            reduce_dtype = get_torch_dtype(precision)
            buffer_dtype = get_torch_dtype(precision)
        else:
            raise ValueError(f"Unable to interpret mixed_precision={mixed_precision}")
    else:
        raise ValueError(f"Unable to interpret mixed_precision={mixed_precision}")

    mixed_precision = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
        keep_low_precision_grads=keep_low_precision_grads,
    )

    return mixed_precision, param_dtype, reduce_dtype, buffer_dtype


class FSDPStrategy(Strategy):
    def __init__(
        self,
        precision="amp_bf16",
        forward_prefetch=True,
        limit_all_gathers=True,
        sync_module_states=True,
        activation_checkpointing=False,
        use_orig_params=False,
    ) -> None:
        self.precision = precision
        self.forward_prefetch = forward_prefetch
        self.limit_all_gathers = limit_all_gathers
        self.sync_module_states = sync_module_states
        self.activation_checkpointing = activation_checkpointing
        self.use_orig_params = use_orig_params

    def setup_model_and_optimizer(self, model, optimizer=None):
        def __auto_wrap_policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
            if recurse:
                return True
            should_be_wrapped = False
            if hasattr(module, "_fsdp_wrap"):
                should_be_wrapped = bool(module._fsdp_wrap)
            else:
                should_be_wrapped = False
            return should_be_wrapped

        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=[model.decoder_cls],
        )

        mixed_precision = get_mixed_precision(self.precision)[0]
        model = FullyShardedDataParallel(
            model,
            auto_wrap_policy=auto_wrap_policy,
            # auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            forward_prefetch=self.forward_prefetch,
            limit_all_gathers=self.limit_all_gathers,
            mixed_precision=mixed_precision,
            sync_module_states=self.sync_module_states,
            use_orig_params=self.use_orig_params,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        )

        if self.activation_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                apply_activation_checkpointing,
                checkpoint_wrapper,
            )

            def check_fn(module: torch.nn.Module) -> bool:
                should_be_check = False
                if hasattr(module, "_activation_checkpoint_wrap"):
                    should_be_check = bool(module._activation_checkpoint_wrap)
                else:
                    should_be_check = False
                return should_be_check

            def check_fn(module: torch.nn.Module) -> bool:
                return isinstance(module, model.decoder_cls)

            wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.REENTRANT,
            )
            apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)

        return model, optimizer
