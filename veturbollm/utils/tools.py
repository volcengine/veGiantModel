import torch


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {}".format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += " | reserved: {}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max reserved: {}".format(torch.cuda.max_memory_reserved() / mega_bytes)
    print("[Rank {}] {}".format(torch.distributed.get_rank(), string), flush=True)


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)
