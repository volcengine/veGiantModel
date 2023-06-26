import pickle
from functools import update_wrapper

import torch
from typing import Mapping


def is_torch_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def is_namedtuple(data):
    """
    Checks if `x` is a `namedtuple` or not. Can have false positives, but only if a user is trying to mimic a
    `namedtuple` perfectly.
    """
    data_type = type(data)
    bases = data_type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(data_type, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(isinstance(member, str) for member in fields)


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple, or namedtuple)
    """
    # Some objects may not be able to instantiate from a generator directly
    if is_namedtuple(obj):
        return type(obj)(*list(generator))
    else:
        return type(obj)(generator)


def recursively_apply(func, data, *args, test_type=is_torch_tensor, error_on_other_type=False, **kwargs):
    """
    Recursively apply a function on a data structure that is a nested list/tuple/dictionary of a given base type.

    Args:
        func (`callable`):
            The function to recursively apply.
        data (nested list/tuple/dictionary of `main_type`):
            The data on which to apply `func`
        *args:
            Positional arguments that will be passed to `func` when applied on the unpacked data.
        main_type (`type`, *optional*, defaults to `torch.Tensor`):
            The base type of the objects to which apply `func`.
        error_on_other_type (`bool`, *optional*, defaults to `False`):
            Whether to return an error or not if after unpacking `data`, we get on an object that is not of type
            `main_type`. If `False`, the function will leave objects of types different than `main_type` unchanged.
        **kwargs:
            Keyword arguments that will be passed to `func` when applied on the unpacked data.

    Returns:
        The same data structure as `data` with `func` applied to every object of type `main_type`.
    """
    if isinstance(data, (tuple, list)):
        return honor_type(
            data,
            (
                recursively_apply(
                    func, o, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for o in data
            ),
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {
                k: recursively_apply(
                    func, v, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for k, v in data.items()
            }
        )
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Unsupported types ({type(data)}) passed to `{func.__name__}`. Only nested list/tuple/dicts of "
            f"objects that are valid for `{test_type.__name__}` should be passed."
        )
    return data


def convert_to_fp32(tensor):
    """
    Recursively converts the elements nested list/tuple/dictionary of tensors in FP16/BF16 precision to FP32.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to convert from FP16/BF16 to FP32.

    Returns:
        The same data structure as `tensor` with all tensors that were in FP16/BF16 precision converted to FP32.
    """

    def _convert_to_fp32(tensor):
        return tensor.float()

    def _is_fp16_bf16_tensor(tensor):
        return hasattr(tensor, "dtype") and (tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16)

    return recursively_apply(_convert_to_fp32, tensor, test_type=_is_fp16_bf16_tensor)


class ConvertOutputsToFp32:
    """
    Decorator to apply to a function outputing tensors (like a model forward pass) that ensures the outputs in FP16
    precision will be convert back to FP32.

    Args:
        model_forward (`Callable`):
            The function which outputs we want to treat.

    Returns:
        The same function as `model_forward` but with converted outputs.
    """

    def __init__(self, model_forward):
        self.model_forward = model_forward
        update_wrapper(self, model_forward)

    def __call__(self, *args, **kwargs):
        return convert_to_fp32(self.model_forward(*args, **kwargs))

    def __getstate__(self):
        raise pickle.PicklingError(
            "Cannot pickle a prepared model with automatic mixed precision, please unwrap the model before pickling it."
        )


def convert_outputs_to_fp32(model_forward):
    model_forward = ConvertOutputsToFp32(model_forward)

    def forward(*args, **kwargs):
        return model_forward(*args, **kwargs)

    # To act like a decorator so that it can be popped when doing `extract_model_from_parallel`
    forward.__wrapped__ = model_forward

    return forward
