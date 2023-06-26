# toodo: move to veturbo
import torch
import torch.distributed as dist


def is_local_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    else:
        return True


class RankPriorityContextManager:
    def __init__(self, priority_rank: int, local_rank: int, enable: bool = True):
        self.priority_rank = priority_rank
        self.local_rank = local_rank
        self.enable = enable

    def __enter__(self):
        if self.enable:
            if self.local_rank != self.priority_rank:
                dist.barrier()

    def __exit__(self, *args):
        if self.enable:
            if self.local_rank == self.priority_rank:
                dist.barrier()


def main_process_first(priority_rank: int = 0):
    """
    When loading the model or dataset, some libraries such as huggingface/transformers
    will download some files first. But in DDP mode, we will use multi-processes to
    launch our program. It makes the download action repeat several times. We use this
    function to make the main process run first, after it is finished the others will
    go on running.

    Example:
    ```python
    with veml.distributed.main_process_first():
        datasets = load_datasets()

    ```
    """
    local_rank = -1
    enable = False

    if dist.is_initialized():
        enable = True
        # TODO: is it robust?
        local_rank = dist.get_rank() % torch.cuda.device_count()

    return RankPriorityContextManager(priority_rank, local_rank, enable)


def extract_model_from_parallel(model, keep_fp32_wrapper: bool = True):
    """
    Extract a model from its distributed containers.

    Args:
        model (`torch.nn.Module`):
            The model to extract.
        keep_fp32_wrapper (`bool`, *optional*):
            Whether to remove mixed precision hooks from the model.

    Returns:
        `torch.nn.Module`: The extracted model.
    """
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)

    while isinstance(model, options):
        model = model.module

    if not keep_fp32_wrapper:
        forward = getattr(model, "forward")
        original_forward = model.__dict__.pop("_original_forward", None)
        if original_forward is not None:
            while hasattr(forward, "__wrapped__"):
                forward = forward.__wrapped__
                if forward == original_forward:
                    break
            model.forward = forward
        if getattr(model, "_converted_to_transformer_engine", False):
            from veturbollm.utils.transformer_engine import convert_model

            convert_model(model, to_transformer_engine=False)
    return model
