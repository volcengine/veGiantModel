import torch
from veturbollm.utils.tools import print_rank_0, report_memory
from veturbollm.global_vars import (
    get_num_microbatches,
    get_tensorboard_writer,
    get_timers,
    get_args,
)


def training_log(
    loss_dict,
    total_loss_dict,
    learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm=None,
    params_norm=None,
    num_zeros_in_grad=None,
):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    total_tflops = loss_dict.pop("tflops", None)

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = "advanced iterations"
    skipped_iters_key = "skipped iterations"
    nan_iters_key = "nan iterations"
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float("inf") or value == -float("inf") or value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        "forward-backward",
        "forward-compute",
        "backward-compute",
        "batch-generator",
        "forward-recv",
        "forward-send",
        "backward-recv",
        "backward-send",
        "forward-send-forward-recv",
        "forward-send-backward-recv",
        "backward-send-forward-recv",
        "backward-send-backward-recv",
        "forward-backward-send-forward-backward-recv",
        "layernorm-grads-all-reduce",
        "embedding-grads-all-reduce",
        "grads-all-reduce",
        "grads-reduce-scatter",
        "params-all-gather",
        "optimizer-copy-to-main-grad",
        "optimizer-unscale-and-check-inf",
        "optimizer-clip-main-grad",
        "optimizer-count-zeros",
        "optimizer-inner-step",
        "optimizer-copy-main-to-model-params",
        "optimizer",
    ]

    # Calculate batch size.
    batch_size = args.training.micro_batch_size * args.distributed.data_parallel_size * get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.logging.log_timers_to_tensorboard and (iteration % args.logging.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration, normalizer=total_iterations)
    if writer and (iteration % args.logging.tensorboard_log_interval == 0):
        if args.logging.log_learning_rate_to_tensorboard:
            writer.add_scalar("learning-rate", learning_rate, iteration)
            writer.add_scalar("learning-rate vs samples", learning_rate, args.consumed_train_samples)
        if args.logging.log_batch_size_to_tensorboard:
            writer.add_scalar("batch-size", batch_size, iteration)
            writer.add_scalar("batch-size vs samples", batch_size, args.consumed_train_samples)
        for key in loss_dict:
            writer.add_scalar(key, loss_dict[key], iteration)
            writer.add_scalar(key + " vs samples", loss_dict[key], args.consumed_train_samples)
        if args.logging.log_loss_scale_to_tensorboard:
            writer.add_scalar("loss-scale", loss_scale, iteration)
            writer.add_scalar("loss-scale vs samples", loss_scale, args.consumed_train_samples)
        if args.logging.log_world_size_to_tensorboard:
            writer.add_scalar("world-size", args.world_size, iteration)
            writer.add_scalar("world-size vs samples", args.world_size, args.consumed_train_samples)
        if grad_norm is not None:
            writer.add_scalar("grad-norm", grad_norm, iteration)
            writer.add_scalar("grad-norm vs samples", grad_norm, args.consumed_train_samples)
        if num_zeros_in_grad is not None:
            writer.add_scalar("num-zeros", num_zeros_in_grad, iteration)
            writer.add_scalar("num-zeros vs samples", num_zeros_in_grad, args.consumed_train_samples)
        if params_norm is not None:
            writer.add_scalar("params-norm", params_norm, iteration)
            writer.add_scalar("params-norm vs samples", params_norm, args.consumed_train_samples)
        if args.logging.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )

    if iteration % args.training.log_interval == 0:
        elapsed_time = timers("interval-time").elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations
        throughput = batch_size / elapsed_time_per_iteration * args.dataset.block_size
        if writer:
            if args.logging.log_timers_to_tensorboard:
                writer.add_scalar("iteration-time", elapsed_time_per_iteration, iteration)
                writer.add_scalar("throughput", throughput, iteration)
        log_string = " iteration {:8d}/{:8d} |".format(iteration, args.training.train_iters)
        log_string += " consumed samples: {:12d} |".format(args.consumed_train_samples)
        log_string += " elapsed time per iteration (ms): {:.1f} |".format(elapsed_time_per_iteration * 1000.0)
        log_string += " learning rate: {:.3E} |".format(learning_rate)
        log_string += " global batch size: {:5d} |".format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                avg = total_loss_dict[key].item() / float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += " {}: {:.6E} |".format(key, avg)
                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        log_string += " loss scale: {:.1f} |".format(loss_scale)
        if grad_norm is not None:
            log_string += " grad norm: {:.3f} |".format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += " num zeros: {:.1f} |".format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += " params norm: {:.3f} |".format(params_norm)
        log_string += " number of skipped iterations: {:3d} |".format(total_loss_dict[skipped_iters_key])
        log_string += " number of nan iterations: {:3d} |".format(total_loss_dict[nan_iters_key])
        log_string += " throughput (tokens/s): {:.0f} |".format(throughput)
        log_string += " tflops: {:.2f} |".format(total_tflops / elapsed_time_per_iteration)
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_0(log_string)
        if report_memory_flag and learning_rate > 0.0:
            # Report memory after optimizer state has been initialized.
            report_memory("(after {} iterations)".format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.training.log_interval)

    return report_memory_flag
