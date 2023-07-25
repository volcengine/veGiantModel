import torch
import warnings

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
