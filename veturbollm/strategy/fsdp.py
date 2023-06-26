import functools
import warnings

import torch
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .base import Strategy


def get_torch_dtype(dtype: str):
    """Convert common string representations of dtypes to torch dtypes."""
    if dtype in ["float32", "torch.float32", "fp32"]:
        return torch.float32
    elif dtype in ["float16", "torch.float16", "half", "fp16", "amp", "amp_fp16"]:
        return torch.float16
    elif dtype in ["bfloat16", "bfloat", "torch.bfloat16", "bf16", "amp_bf16"]:
        return torch.bfloat16
    elif dtype in ["float8", "torch.float8", "fp8", "amp_fp8"]:
        if hasattr(torch, "float8"):
            raise NotImplementedError("Torch has enabled float8. This should be updated to `return torch.float8`")
        else:
            warnings.warn("We use torch.bfloat16 by default for amp_fp8 as there is no fp8 datatype in PyTorch yet.")
            return torch.bfloat16
    else:
        raise ValueError(f"Not sure how to convert dtype={dtype} to a torch dtype.")


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
        self, precision="amp_bf16", forward_prefetch=True, limit_all_gathers=True, sync_module_states=True
    ) -> None:
        self.precision = precision
        self.forward_prefetch = forward_prefetch
        self.limit_all_gathers = limit_all_gathers
        self.sync_module_states = sync_module_states

    def setup_model_and_optimizer(self, model, optimizer=None):
        if "gpt2" in model.config.model_type:
            from veturbollm.modules.block import Block

            transformer_layer_cls = [Block]
        elif "llama" in model.config.model_type:
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer

            transformer_layer_cls = [LlamaDecoderLayer]
        else:
            raise ValueError(f"Unsupported model {model._name_or_path}")

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_cls,
        )

        mixed_precision = get_mixed_precision(self.precision)[0]

        model = FullyShardedDataParallel(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            forward_prefetch=self.forward_prefetch,
            limit_all_gathers=self.limit_all_gathers,
            mixed_precision=mixed_precision,
            sync_module_states=self.sync_module_states,
        )

        return model, optimizer
