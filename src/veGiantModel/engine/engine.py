# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
# Copyright 2019 The Microsoft DeepSpeed Team
import os

from types import MethodType

import torch

import torch.distributed as dist

from deepspeed.utils.logging import logger
from deepspeed.utils.timer import ThroughputTimer

from deepspeed.runtime.engine import MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.runtime.dataloader import RepeatingLoader

from deepspeed.runtime.pipe.module import PipelineModule, PipelineError
from deepspeed.runtime.pipe.engine import PipelineEngine
from util.checkpoint_util import load_megatron_model_state, get_ckpt_name

from . import p2p
from . import schedule
try:
    import byteps.torch as bps
except ImportError:
    print("byteps is not installed. Pipeline parallelism is disabled")
    bps = None

from .module import VeGiantModule
from deepspeed.utils import log_dist
import logging
from torch._six import inf

LOG_STAGE = -2
DATA_PARALLEL_ID = -2

try:
    from apex import amp
except ImportError:
    # Fail silently so we don't spam logs unnecessarily if user isn't using amp
    pass


def is_even(number):
    return number % 2 == 0

ENABLE_PYTORCH_BROADCAST = os.environ.get("ENABLE_PYTORCH_BROADCAST", "0") != "0"



DS_PIPE_VERBOSE = int(os.environ.get('DS_PIPE_VERBOSE', "0"))
MEGATRON_DEBUG_DATA = os.environ.get('MEGATRON_DEBUG_DATA', "0") != "0"
MEGATRON_DEBUG_GRAD = os.environ.get('MEGATRON_DEBUG_GRAD', "0") != "0"
ENABLE_BPS_PARTITION = os.environ.get("ENABLE_BPS_PARTITION", "0") != "0"


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()

def _dtype_to_code(dtype):
    if dtype == torch.half:
        return 0
    elif dtype == torch.float:
        return 1
    elif dtype == torch.int16:
        return 2
    elif dtype == torch.int32:
        return 3
    elif dtype == torch.int64:
        return 4
    elif dtype == torch.bool:
        return 5
    else:
        raise AssertionError("not recognized tensor type for pipeline send")

def _code_to_dtype(code):
    if code == 0:
        return torch.half
    elif code == 1:
        return torch.float
    elif code == 2:
        return torch.int16
    elif code == 3:
        return torch.int32
    elif code == 4:
        return torch.int64
    elif code == 5:
        return torch.bool
    else:
        raise AssertionError("not recognized tensor type code for pipeline recv")

class VeGiantModelEngine(PipelineEngine):
    """ A training engine hybrid pipeline, data, and model parallel training.

    This engine is created by ``deepspeed.initialize()`` when a :class:`PipelineModule`
    is provided.
    """
    def overwrite(self, config_params, args):
        if args.batch_size is not None:
            log_dist(f'overwrite dsconfig train_micro_batch_size_per_gpu to {args.batch_size}', \
                ranks=[-1], level=logging.DEBUG)
            config_params['train_micro_batch_size_per_gpu'] = args.batch_size
        
        if args.gradient_accumulation_steps is not None:
            log_dist(f'overwrite dsconfig gradient_accumulation_steps to {args.gradient_accumulation_steps}', \
                ranks=[-1], level=logging.DEBUG)
            config_params['gradient_accumulation_steps'] = args.gradient_accumulation_steps

        if args.train_batch_size is not None:
            log_dist(f'overwrite dsconfig train_batch_size to {args.train_batch_size}, ', \
                ranks=[-1], level=logging.DEBUG)
            config_params['train_batch_size'] = args.train_batch_size

        if args.log_interval is not None:
            config_params['steps_per_print'] = args.log_interval

    def __init__(self, args,
                    model,
                    optimizer,
                    model_parameters,
                    training_data,
                    lr_scheduler,
                    mpu,
                    dist_init_required,
                    collate_fn,
                    config_params):
        
        self.overwrite(config_params, args)
        super(PipelineEngine, self).__init__(args,
                    model,
                    optimizer,
                    model_parameters,
                    training_data,
                    lr_scheduler,
                    mpu,
                    dist_init_required,
                    collate_fn,
                    config_params)
        assert isinstance(self.module, PipelineModule), "model must base PipelineModule"

        # pipeline step for logging
        self.args = args
        self.log_batch_step_id = -1
        self.train_mode = True

        self.enable_backward_allreduce = False
        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()
        self.first_train = True
        self.first_eval = True

        # Set Grid and Communication Groups
        self.grid = self.module._grid
        if self.grid.get_global_rank() == 0:
            logger.info(f'CONFIG: micro_batches={self.micro_batches} '
                        f'micro_batch_size={self.micro_batch_size}')

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        assert self.train_batch_size() == \
            self.micro_batch_size * self.micro_batches * self.grid.data_parallel_size

        #  Set Stage Inf
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.mp_id = self.grid.get_model_parallel_id()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        self.data_iterator = None
        self.batch_fn = None
        self.result_dict = {}

        self._force_grad_boundary = False

        self.batch_timer = ThroughputTimer(batch_size=self.micro_batch_size *
                                           self.micro_batches,
                                           num_workers=self.dp_world_size,
                                           logging_fn=self.tput_log,
                                           monitor_memory=False,
                                           steps_per_output=self.steps_per_print())

        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses
        if self.training_data:
            self._build_data_iter(self.training_data)

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        # Partition input/output buffers
        self.is_pipe_partitioned = False if self.args.broadcast_activation else (self.is_model_parallel and ENABLE_PYTORCH_BROADCAST)
        self.is_grad_partitioned = False

        model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        unique_params = num_params
        # Subtract tied parameters if we don't own them
        if self.module.tied_comms:
            tied_params = 0
            for key, d in self.module.tied_comms.items():
                if self.global_rank != min(d['ranks']):
                    tied_params += sum(p.numel() for p in d['module'].parameters())
            unique_params -= tied_params
        params_tensor = torch.LongTensor(data=[num_params,
                                               unique_params]).to(self.device)
        print(f'Calculating param sizes ... ', flush=True)


        dist.all_reduce(params_tensor, group=self.grid.get_model_parallel_group())
        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]
        if self.grid.data_parallel_id == 0:
            logger.info(f'RANK={self.global_rank} '
                        f'STAGE={self.stage_id} '
                        f'LAYERS={self.module._local_stop - self.module._local_start} '
                        f'[{self.module._local_start}, {self.module._local_stop}) '
                        f'STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) '
                        f'TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) '
                        f'UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M)')

        print(f'DONE calculating param sizes. Now init proc groups', flush=True)

        #intialize peer-2-peer communication and allreduce groups
        if self.is_pipe_parallel:
            p2p.init_process_groups(self.grid)

        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            'inputs' : [],   # batch input and received activations
            'labels' : [],   # labels from batch input
            'outputs' : [],  # activations
            'output_tensors' : [], # tensor object to preserve backward graph
            'bps_act_recv' : [],  # activations recv
            'bps_grad_recv' : [],  # activations recv
        }
        self.pipe_recv_buf = None
        self.grad_layer = None

        self.meta_buffer = None

        self.first_output_send = True
        self.first_gradient_send = True

        #stores the loss for the current micro batch being processed
        self.loss = torch.tensor(0.0).to(self.device)
        self.metric = 0

        #stores the loss for the entire batch
        self.total_loss = None
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        self.dp_group_loss = torch.tensor(0.0, requires_grad=False).to(self.device)

        if self._config.pipeline['activation_checkpoint_interval'] > 0:
            self.module.activation_checkpoint_interval = self._config.pipeline[
                'activation_checkpoint_interval']

        if self.is_last_stage():
            self.loss_model = self.module.loss_fn

        log_dist(f'Initialize pipeline communicators', \
            ranks=[-1], level=logging.DEBUG)

        # Initialize pipeline communicators. Just send a 0.
        if is_even(self.stage_id):
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
        else:
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)
        
        log_dist(f'DONE Initialize pipeline communicators', \
            ranks=[-1], level=logging.DEBUG)

        # XXX look into timer reporting timing
        # Initialize some timers because of early weirdness.
        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
            self.timers('forward_microstep').stop()
            self.timers('backward_microstep').start()
            self.timers('backward_microstep').stop()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward_allreduce_microstep').start()
            self.timers('backward_allreduce_microstep').stop()
            self.timers('backward_allreduce').start()
            self.timers('backward_allreduce').stop()
            self.timers('step_microstep').start()
            self.timers('step_microstep').stop()

        if self.local_rank == -1:
            # or number of visiable device will be better
            self.local_rank = self.global_rank % torch.cuda.device_count()

        if not p2p.ENABLE_PYTORCH_BROADCAST:
            gpu_per_node = int(os.environ['GPU_PER_WORKER'])
            print(f'bps init worker: {gpu_per_node}, {self.local_rank}/{self.global_rank}', flush=True)
            os.environ['BYTEPS_LOCAL_RANK'] = str(self.local_rank)
            os.environ['BYTEPS_LOCAL_SIZE'] = str(gpu_per_node)
            os.environ['BYTEPS_VISIBLE_DEVICE'] = str(self.local_rank)
            os.environ['DMLC_ROLE'] = 'joint'
            os.environ['DMLC_WORKER_ID'] = str(self.global_rank)
            bps.init(lazy=False)
            print(f'bps init DONE', flush=True)
        self._read_in_all_rank = False


    def _profiling_func_exit(self):
        torch.cuda.nvtx.range_pop()
    
    def _profiling_func_enter(self, func):
        torch.cuda.nvtx.range_push(f'stage_id: {self.stage_id}, mp_id: {self.mp_id}, fun: {func}')

    def _build_data_iter(self, dataset):
        if not isinstance(dataset, torch.utils.data.Dataset):
            self.set_dataloader(dataset)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.dp_world_size,
                rank=self.mpu.get_data_parallel_rank(),
                shuffle=False)
            # Build a loader and make it repeating.
            pipe_dataloader = self.deepspeed_io(dataset, data_sampler=sampler)
            pipe_dataloader = RepeatingLoader(pipe_dataloader)
            self.set_dataloader(pipe_dataloader)

    def _exec_reduce_tied_grads(self):
        self._profiling_func_enter('_exec_reduce_tied_grads')
        self.module.allreduce_tied_weight_gradients()
        self._profiling_func_exit()

    def _exec_reduce_grads(self):
        self._profiling_func_enter('_exec_reduce_grads')
        self._force_grad_boundary = True
        if self.is_data_parallel:
            self.buffered_allreduce_fallback(
                elements_per_buffer=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False
        self._profiling_func_exit()


    def _reserve_pipe_buffers(self, num_buffers):
        """Ensure that each pipeline buffer has at least ``num_buffers`` slots.

        This method only reserves slots and does not allocate tensors.

        Args:
            num_buffers (int): The number of buffers to reserve.
        """
        if self.num_pipe_buffers >= num_buffers:
            return

        num_added = num_buffers - self.num_pipe_buffers
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * num_added)
        self.num_pipe_buffers = num_buffers

    def train_batch(self, data_iter=None):
        """Progress the pipeline to train the next batch of data. The engine will ingest
        ``self.train_batch_size()`` total samples collectively across all workers.


        An iterator that over training data should be provided as an argument
        unless ``deepspeed.initialize()`` was provided a training set. In that event,
        the training data will automatically be read.


        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator, optional): Iterator of training data.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """

        if DS_PIPE_VERBOSE:
            print(f'[{self.global_rank}] start train_batch()', flush=True)
        if not torch._C.is_grad_enabled():
            raise RuntimeError(
                f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        if data_iter is not None:
            self.set_dataiterator(data_iter)

        self.module.train()
        self.train()
        self.total_loss = None
        self._compute_loss = True

        # Do the work
        self.timers('train_batch').start()
        # We only enable prefetching starting from the second batch
        if not ENABLE_PYTORCH_BROADCAST:
            sched = schedule.BytePSTrainSchedule(micro_batches=self.micro_batches,
                                                stages=self.num_stages,
                                                stage_id=self.stage_id, prefetch=not self.first_train)
        else:
            sched = schedule.TrainSchedule(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        cmd = ','.join(str(x) for x in sched)
        # log_dist(f'stage_id: {self.stage_id}, sched:{cmd}', ranks=[-1], level=logging.INFO)
        self._exec_schedule(sched)
        self.agg_train_loss = self._aggregate_total_loss()
        self.timers('train_batch').stop()

        if self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                elapsed = self.timers('train_batch').elapsed(reset=True)
                iter_time = elapsed / self.steps_per_print()
                tput = self.train_batch_size() / iter_time
                print(f'steps: {self.global_steps} '
                      f'loss: {self.agg_train_loss:0.4f} '
                      f'iter time (s): {iter_time:0.3f} '
                      f'samples/sec: {tput:0.3f}')

        # Tensorboard
        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/train_loss',
                                        self.agg_train_loss.mean().item(),
                                        self.global_samples)]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])
                if self.global_steps % self.steps_per_print() == 0:
                    self.summary_writer.flush()

        if self.wall_clock_breakdown(
        ) and self.global_steps % self.steps_per_print() == 0:
            self.timers.log([
                'pipe_send_output',
                'pipe_send_grad',
                'pipe_recv_input',
                'pipe_recv_grad'
            ])

        # TODO: should return precisely what loss returned and allow others to be queried?
        self.first_train = False
        if DS_PIPE_VERBOSE:
            print(f'[{self.global_rank}] DONE train_batch()', flush=True)
        
        self.result_dict['loss'] = self.agg_train_loss
        return self.result_dict

    def eval_batch(self, data_iter, compute_loss=True, reduce_output='avg', read_in_all_rank=False):
        """Evaluate the pipeline on a batch of data from ``data_iter``. The
        engine will evaluate ``self.train_batch_size()`` total samples
        collectively across all workers.

        This method is equivalent to:

        .. code-block:: python

            module.eval()
            with torch.no_grad():
                output = module(batch)

        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator): Iterator of data to evaluate.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """

        self.module.eval()
        self.eval()
        self.total_loss = None
        eval_output = None
        self._compute_loss = compute_loss
        self._read_in_all_rank = read_in_all_rank

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # Do the work
        self.timers('eval_batch').start()
        if not ENABLE_PYTORCH_BROADCAST:
            sched = schedule.BytePSInferenceSchedule(micro_batches=1,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id, prefetch=False)
        else:
            sched = schedule.InferenceSchedule(micro_batches=self.micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)
        cmd = ','.join(str(x) for x in sched)
        # log_dist(f'stage_id: {self.stage_id}, sched:{cmd}', ranks=[-1], level=logging.INFO)
        with torch.no_grad():
            self._exec_schedule(sched)

        if self.is_last_stage():
            eval_output = self._reduce_outputs(self.fwd_outputs, reduce=reduce_output)

        if compute_loss:
            eval_output = self._bcast_pipe_scalar(eval_output)

        self.timers('eval_batch').stop()

        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/eval_loss',
                                        eval_output.mean().item(),
                                        self.global_samples)]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])
                self.summary_writer.flush()

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        # Reset any buffers that may have been populated during the forward passes.
        self.first_eval = False
        return eval_output

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1

    def _reduce_outputs(self, outputs, reduce='avg', reduce_dp=True):
        if reduce is None:
            return outputs

        if reduce.lower() == 'avg':
            # first sum over all microbatches
            if torch.is_tensor(outputs[0]):
                reduced = sum(outputs)
            else:
                assert isinstance(outputs, (list, tuple))
                reduced = [torch.zeros_like(o) for o in outputs[0]]
                for idx, out in outputs:
                    reduced[idx] += out

            # Average over the microbatches
            reduced = self._scale_loss(reduced)

            # Average over DP groups
            if reduce_dp and self.is_data_parallel:
                if torch.is_tensor(reduced):
                    dist.all_reduce(reduced, group=self.mpu.get_data_parallel_group())
                    reduced /= self.dp_world_size
                else:
                    for idx in range(len(reduced)):
                        dist.all_reduce(reduced[idx],
                                        group=self.mpu.get_data_parallel_group())
                        reduced[idx] /= self.dp_world_size

            return reduced
        else:
            raise NotImplementedError(f'reduction type {reduce} not supported.')

    def _bcast_pipe_scalar(self, data, src_rank=None, dtype=torch.float32):
        # Default to last stage (e.g., for broadcasting loss)
        if src_rank is None:
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
        assert src_rank in self.grid.pp_group

        if self.global_rank == src_rank:
            result = data.clone().detach()
        else:
            result = torch.Tensor([0.]).type(dtype).to(self.device)

        dist.broadcast(tensor=result,
                       src=src_rank,
                       group=self.mpu.get_pipe_parallel_group())

        return result

    def _aggregate_metric(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if self.is_last_stage():
            if DS_PIPE_VERBOSE:
                print(f'[{self.global_rank}] bcast src={self.global_rank} group={self.grid.pp_group}', flush=True)
            if self.is_data_parallel:
                assert False

            assert self.global_rank in self.grid.pp_group
            metric = torch.Tensor([self.metric]).to(self.device)
            dist.broadcast(tensor=metric,
                           src=self.global_rank,
                           group=self.mpu.get_pipe_parallel_group())

        else:
            # Get loss from last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            if DS_PIPE_VERBOSE:
                print(f'[{self.global_rank}] bcast src={src_rank} group={self.grid.pp_group}', flush=True)
            assert src_rank in self.grid.pp_group
            metric = torch.Tensor([0.]).to(self.device)
            dist.broadcast(tensor=metric,
                           src=src_rank,
                           group=self.grid.get_pipe_parallel_group())
            self.metric = metric.clone().detach().cpu().numpy()

        return self.metric

    def _aggregate_total_loss(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if self.is_last_stage():
            # XXX Hack: do not scale loss
            loss = self._scale_loss(self.total_loss)

            self.dp_group_loss = loss.clone().detach()

            ## Average loss across all data-parallel groups
            agg_loss = self.dp_group_loss.clone().detach()

            if DS_PIPE_VERBOSE:
                print(f'[{self.global_rank}] bcast SENDER src={self.global_rank} group={self.grid.pp_group}', flush=True)
            if self.is_data_parallel:
                dist.all_reduce(agg_loss, group=self.mpu.get_data_parallel_group())
                agg_loss /= self.dp_world_size

            assert self.global_rank in self.grid.pp_group
            losses = torch.Tensor([self.dp_group_loss, agg_loss]).to(self.device)
            dist.broadcast(tensor=losses,
                           src=self.global_rank,
                           group=self.mpu.get_pipe_parallel_group())

        else:
            # Get loss from last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            losses = torch.Tensor([0., 0.]).to(self.device)
            if DS_PIPE_VERBOSE:
                print(f'[{self.global_rank}] bcast RECVER src={src_rank} group={self.grid.pp_group}', flush=True)
            dist.broadcast(tensor=losses,
                           src=src_rank,
                           group=self.grid.get_pipe_parallel_group())
            self.dp_group_loss = losses[0].clone().detach()
            agg_loss = losses[1].clone().detach()
        if DS_PIPE_VERBOSE:
            print(f'DONE aggregate total loss', flush=True)
        return agg_loss

    def set_dataloader(self, loader):
        """"""
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = loader
            self.data_iterator = iter(self.training_dataloader)

    def set_dataiterator(self, iterator):
        """ Store an iterator to sample for training data. """
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = None
            self.data_iterator = iterator

    def set_batch_fn(self, fn):
        self.batch_fn = fn

    def is_gradient_accumulation_boundary(self):
        """True if the engine is executing a gradient reduction or optimizer step instruction.

        This is overridden from :class:`DeepSpeedEngine` to force reductions
        and steps when the pipeline engine is instructed to do so.

        Returns:
            bool: whether reductions and optimizer steps should occur.
        """
        return self._force_grad_boundary


    def tput_log(self, *msg):
        if self.global_rank == 0 and self.global_steps % self.steps_per_print() == 0:
            print(*msg)

    def _next_batch(self):
        if self.is_model_parallel:
            mp_rank = self.grid.get_slice_parallel_rank()
        else:
            mp_rank = 0

        batch = None
        # Only MP rank 0 loads the data.
        if self._read_in_all_rank or mp_rank == 0:
            if self.data_iterator is None:
                raise ValueError(f"RANK={self.global_rank} no data iterator provided.")
            batch = next(self.data_iterator)

        # All MP ranks participate in batch_fn, where they might broadcast the data.
        if self.batch_fn:
            batch = self.batch_fn(batch, self.train_mode)

        # Sanity check dimensions.
        # XXX: the last minibatch with size < micro_batch_size kills us
        if torch.is_tensor(batch[0]):
            if batch[0].size(0) != self.micro_batch_size:
                print(f'size mismatch: {batch[0].size(0)} mb: {self.micro_batch_size}')
                assert batch[0].size(0) == self.micro_batch_size
                return self._next_batch()
        else:
            assert torch.is_tensor(batch[0][0])
            if batch[0][0].size(0) != self.micro_batch_size:
                print(f'HB next_batch: {batch[0][0].shape} vs {self.micro_batch_size}', flush=True)
                return self._next_batch()
        
        return batch

    def _exec_bps_forward_pass(self, buffer_id):
        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)
        self._profiling_func_enter('_exec_bps_forward_pass')

        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()

        # collect the partitioned input from the previous stage
        assert not self.is_pipe_partitioned

        # Zero out the gradients each time we use the tensor because only the data in
        # tensor changes across batches
        self._zero_grads(inputs)

        outputs = super(PipelineEngine, self).forward(inputs)

        # Partition the outputs if we are not the last stage
        assert not self.is_pipe_partitioned

        self.pipe_buffers['outputs'][buffer_id] = outputs

        # Optionally compute loss and metrics on the last device
        if self.is_last_stage():
            if self._compute_loss and self.loss_model is not None:
                labels = self.pipe_buffers['labels'][buffer_id]
                ret = self.loss_model(outputs, labels)
                if isinstance(ret, dict):
                    self.result_dict = ret
                    self.loss = self.result_dict['loss']
                else:
                    self.loss = ret
            else:
                # Some models just return loss from forward()
                self.loss = outputs
            # get metric from self.module

            if isinstance(self.loss, torch.Tensor):
                self.fwd_outputs.append(self.loss.detach())
                if self.total_loss is None:
                    self.total_loss = torch.zeros_like(self.loss)
                self.total_loss += self.loss.detach()
            else:
                self.fwd_outputs.append([l.detach() for l in self.loss])

                if self.total_loss is None:
                    self.total_loss = [torch.zeros_like(l) for l in self.loss]
                for idx, l in enumerate(self.loss):
                    self.total_loss[idx] += l.detach()

        self._profiling_func_exit()

    def _exec_bps_backward_pass(self, buffer_id):
        self._profiling_func_enter('_exec_bps_backward_pass')
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        self.mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage():
            super(PipelineEngine, self).backward(self.loss)
            self.mem_status('AFTER BWD')
            self._profiling_func_exit()
            return

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.wall_clock_breakdown():
            self.timers('backward_microstep').start()
            self.timers('backward').start()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner').start()

        assert not self.is_pipe_partitioned
        assert not self.is_grad_partitioned
        # TODO: do we need to clone()?
        grad_tensors = self.pipe_buffers['bps_grad_recv'][buffer_id]

        if isinstance(outputs, tuple):
            out_tensors = [t for t in outputs if t.is_floating_point()]
            assert len(out_tensors) == len(grad_tensors)
            new_out_tensors=[]
            new_grad_tensors=[]
            for t,g in zip(out_tensors, grad_tensors):
                if t.requires_grad:
                    new_out_tensors.append(t)
                    new_grad_tensors.append(g)

            assert len(new_out_tensors) == len(new_grad_tensors)
            torch.autograd.backward(tensors=new_out_tensors, grad_tensors=new_grad_tensors)
        else:
            torch.autograd.backward(tensors=(outputs,), grad_tensors=(grad_tensors,))

        # Free up the memory from the output of forward()
        self.pipe_buffers['output_tensors'][buffer_id] = None
        self.pipe_buffers['outputs'][buffer_id] = None
        grad_tensors = None

        if self.wall_clock_breakdown():
            self.timers('backward_inner').stop()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward').stop()
            self.timers('backward_microstep').stop()

        self.mem_status('AFTER BWD')
        self._profiling_func_exit()

    def _exec_load_micro_batch(self, buffer_id):
        self._profiling_func_enter('_exec_load_micro_batch')
        if self.wall_clock_breakdown():
            self.timers('batch_input').start()

        if self.is_first_stage():
            batch = self._next_batch()
            loaded = None
            if torch.is_tensor(batch[0]):
                loaded = batch[0].clone().to(self.device).detach()
                loaded.requires_grad = loaded.is_floating_point()
                if MEGATRON_DEBUG_DATA:
                    print(f'batch = {loaded.sum().detach()}', flush=True)
            else:
                assert isinstance(batch[0], tuple)
                # Assume list or tuple
                loaded = []
                for x in batch[0]:
                    assert torch.is_tensor(x)
                    mine = x.clone().detach().to(self.device)
                    mine.requires_grad = mine.is_floating_point()
                    loaded.append(mine)
                loaded = tuple(loaded)
                if MEGATRON_DEBUG_DATA:
                    print(f'rank: {self.global_rank}, stage: {self.stage_id},  batch[0] = {[x.sum().detach() for x in loaded]}', flush=True)

            self.pipe_buffers['inputs'][buffer_id] = loaded

        if self.is_last_stage() and self._compute_loss:
            batch = self._next_batch()
            loaded = batch[1]
            if torch.is_tensor(batch[1]):
                loaded = batch[1].to(self.device)
                if MEGATRON_DEBUG_DATA:
                    print(f'rank: {self.global_rank}, stage: {self.stage_id},  batch[1] = {[x.sum().detach() for x in loaded]}', flush=True)
            elif isinstance(batch[1], tuple):
                loaded = []
                for x in batch[1]:
                    assert torch.is_tensor(x)
                    x = x.to(self.device).detach()
                    loaded.append(x)
                loaded = tuple(loaded)
                if MEGATRON_DEBUG_DATA:
                    print(f'rank: {self.global_rank}, stage: {self.stage_id},  batch[1] = {[x.sum().detach() for x in loaded]}', flush=True)

            self.pipe_buffers['labels'][buffer_id] = loaded

        if self.wall_clock_breakdown():
            self.timers('batch_input').stop()
        self._profiling_func_exit()

    def _send_tensor_meta(self, buffer, recv_stage):
        self._profiling_func_enter('_send_tensor_meta')
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape
        """
        send_bytes = 0
        if isinstance(buffer, torch.Tensor):
            type_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            send_shape = torch.LongTensor(data=buffer.size()).to(self.device)
            send_ndims = torch.LongTensor(data=[len(buffer.size())]).to(self.device)
            send_dtype = torch.LongTensor(data=[_dtype_to_code(buffer.dtype)]).to(self.device)
            p2p.send(send_ndims, recv_stage)
            p2p.send(send_shape, recv_stage)
            p2p.send(send_dtype, recv_stage)
            send_bytes += _tensor_bytes(buffer)
        elif isinstance(buffer, list):
            assert (False)
            type_tensor = torch.LongTensor(data=[1]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                send_dtype = torch.LongTensor(data=_dtype_to_code([tensor.dtype])).to(self.device)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                p2p.send(send_dtype, recv_stage)
                send_bytes += _tensor_bytes(tensor)
        elif isinstance(buffer, tuple):
            type_tensor = torch.LongTensor(data=[2]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for idx, tensor in enumerate(buffer):
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                send_dtype = torch.LongTensor(data=[_dtype_to_code(tensor.dtype)]).to(self.device)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                p2p.send(send_dtype, recv_stage)
                # Useful for performance debugging.
                '''
                new_bytes = _tensor_bytes(tensor)
                send_bytes += _tensor_bytes(tensor)
                # Useful for performance debugging.
                if self.grid.data_parallel_id == 0:
                    print(
                        f'STAGE={self.stage_id} pipe-send-volume[{idx}]: shape={send_shape} {new_bytes/1024**2:0.2f}MB'
                    )
                '''
        else:
            raise NotImplementedError(f'Could not send meta type {type(buffer)}')

        self._profiling_func_exit()
        # Useful for performance debugging.
        '''
        if self.grid.data_parallel_id == 0:
            print(f'STAGE={self.stage_id} pipe-send-volume: {send_bytes/1024**2:0.2f}MB')
        '''

    def _recv_tensor_meta(self, send_stage):
        self._profiling_func_enter('_recv_tensor_meta')
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape

        Returns:
            Allocated buffer for receiving from send_stage.
        """

        type_tensor = torch.LongTensor(data=[0]).to(self.device)
        p2p.recv(type_tensor, send_stage)
        recv_type = type_tensor.item()

        # A single tensor will be sent.
        if recv_type == 0:
            recv_ndims = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(recv_ndims, send_stage)
            recv_ndims = recv_ndims.item()
            recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
            p2p.recv(recv_shape, send_stage)
            recv_shape = recv_shape.tolist()
            recv_dtype = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(recv_dtype, send_stage)
            recv_dtype_code = recv_dtype.item()
            recv_dtype = _code_to_dtype(recv_dtype_code)
            return self._allocate_buffer2(recv_shape, recv_dtype, num_buffers=1)[0]

        # List or tuple of tensors
        elif recv_type == 1 or recv_type == 2:
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(count_tensor, send_stage)
            num_tensors = count_tensor.item()
            recv_shapes = []
            recv_dtypes = []
            for idx in range(num_tensors):
                recv_ndims = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_ndims, send_stage)
                recv_ndims = recv_ndims.item()
                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
                p2p.recv(recv_shape, send_stage)
                recv_shapes.append(recv_shape.tolist())
                recv_dtype = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_dtype, send_stage)
                recv_dtype_code = recv_dtype.item()
                recv_dtype = _code_to_dtype(recv_dtype_code)
                recv_dtypes.append(recv_dtype)

            buffers = self._allocate_buffers2(recv_shapes, recv_dtypes, num_buffers=1)[0]
            # Convert to tuples if requested.
            if recv_type == 2:
                buffers = tuple(buffers)
            return buffers

        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')
        self._profiling_func_exit()

    def _mp_slice(self, x):
        mp_size = self.grid.get_model_parallel_world_size()
        return x.reshape((mp_size, -1))[self.mp_id:self.mp_id+1, :].detach()

    def _mp_view(self, x, rank):
        mp_size = self.grid.get_model_parallel_world_size()
        return x.view((mp_size, -1))[rank:rank+1, :]

    def _exec_bps_send_partitioned_activations(self, buffer_id):
        self._profiling_func_enter('_exec_bps_send_activations')
        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.first_output_send:
            self.first_output_send = False
            self._send_tensor_meta(outputs, self.next_stage)

        assert not self.args.broadcast_activation
        assert ENABLE_BPS_PARTITION
        name = f'act_{buffer_id}'
        if isinstance(outputs, torch.Tensor):
            p2p.bps_send(self._mp_slice(outputs.contiguous()),
                         self.next_stage, name, index=0, async_op=True)
        elif isinstance(outputs, (tuple, list)):
            for idx, buffer in enumerate(outputs):
                if DS_PIPE_VERBOSE >= 3:
                    log_dist(f'DS BPS_SEND tensors {idx}/{len(outputs)}, next_stage={self.next_stage} sum={self._mp_slice(buffer.contiguous()).sum()}', ranks=[-1])
                p2p.bps_send(self._mp_slice(buffer.contiguous()), self.next_stage,
                             name, index=idx, async_op=True)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')

        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').stop()
        self._profiling_func_exit()

    def _exec_bps_send_activations(self, buffer_id):
        self._profiling_func_enter('_exec_bps_send_activations')
        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.first_output_send:
            self.first_output_send = False
            self._send_tensor_meta(outputs, self.next_stage)

        assert not self.args.broadcast_activation
        assert not ENABLE_BPS_PARTITION
        if self.mp_id == 0:
            name = f'act_{buffer_id}'
            if isinstance(outputs, torch.Tensor):
                p2p.bps_send(outputs.contiguous(), self.next_stage, name, index=0, async_op=True)
            elif isinstance(outputs, (tuple, list)):
                for idx, buffer in enumerate(outputs):
                    if DS_PIPE_VERBOSE >= 3:
                        log_dist(f'DS BPS_SEND tensors {idx}/{len(outputs)} start', ranks=[-1])
                    p2p.bps_send(buffer.contiguous(), self.next_stage, name, index=idx, async_op=True)
                    if DS_PIPE_VERBOSE >= 3:
                        log_dist(f'DS BPS_SEND tensors {idx}/{len(outputs)} end', ranks=[-1])
            else:
                raise NotImplementedError('Could not send output of type '
                                          f'{type(outputs)}')

        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').stop()
        self._profiling_func_exit()

    def _exec_bps_send_grads(self, buffer_id):
        self._profiling_func_enter('_exec_bps_send_grads')
        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').start()

        inputs = self.pipe_buffers['inputs'][buffer_id]

        # Partition the gradient
        assert not self.is_grad_partitioned
        assert not self.args.broadcast_grads

        name = f'grad_{buffer_id}'
        # only MP rank 0 sends the gradient
        if self.grid.get_model_parallel_rank() == 0:
            if isinstance(inputs, torch.Tensor):
                if inputs.grad is None:
                    send_data = self._allocate_zeros(inputs.size())
                else:
                    send_data = inputs.grad
                assert send_data.is_floating_point()
                assert send_data is not None
                p2p.bps_send(send_data, self.prev_stage, name, index=0, async_op=True)

            else:
                for idx, buffer in enumerate(inputs):
                    if not buffer.is_floating_point():
                        continue
                    if buffer.grad is None:
                        send_data = self._allocate_zeros(buffer.size())
                    else:
                        send_data = buffer.grad
                    assert send_data.is_floating_point()
                    assert send_data is not None
                    p2p.bps_send(send_data, self.prev_stage, name, index=idx, async_op=True)

        # We can free up the input buffer now
        self.pipe_buffers['inputs'][buffer_id] = None

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').stop()
        self._profiling_func_exit()

    def _exec_bps_send_partitioned_grads(self, buffer_id):
        self._profiling_func_enter('_exec_bps_send_grads')
        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').start()

        inputs = self.pipe_buffers['inputs'][buffer_id]

        # Partition the gradient
        assert not self.is_grad_partitioned
        assert not self.args.broadcast_grads
        assert ENABLE_BPS_PARTITION

        name = f'grad_{buffer_id}'
        if isinstance(inputs, torch.Tensor):
            if inputs.grad is None:
                send_data = self._allocate_zeros(inputs.size())
            else:
                send_data = inputs.grad
            assert send_data.is_floating_point()
            assert send_data is not None
            p2p.bps_send(self._mp_slice(send_data), self.prev_stage, name,
                         index=0, async_op=True)
        else:
            for idx, buffer in enumerate(inputs):
                if not buffer.is_floating_point():
                    continue
                if buffer.grad is None:
                    send_data = self._allocate_zeros(buffer.size())
                else:
                    send_data = buffer.grad
                assert send_data.is_floating_point()
                assert send_data is not None
                p2p.bps_send(self._mp_slice(send_data), self.prev_stage,
                             name, index=idx, async_op=True)

        # We can free up the input buffer now
        self.pipe_buffers['inputs'][buffer_id] = None

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').stop()
        self._profiling_func_exit()

    def _exec_bps_sync_all(self):
        p2p.bps_sync_all()

    def _exec_bps_sync_partitioned_grads(self, buffer_id):
        name = f'grad_{buffer_id}'
        recv_buff = self.pipe_buffers['bps_grad_recv'][buffer_id]
        if isinstance(recv_buff, torch.Tensor):
            p2p.bps_sync(self.next_stage, name, index=0)
        else:
            for i in range(len(recv_buff)):
                p2p.bps_sync(self.next_stage, name, index=i)

        # all_gather the gradient from other ranks
        mp_size = self.grid.model_parallel_size
        if mp_size > 1:
            src_rank = self.grid.slice_parallel_src_id
            group = self.grid.slice_proc_group
            if isinstance(recv_buff, torch.Tensor):
                recv_buff_views = [self._mp_view(recv_buff, i) for i in range(mp_size)]
                dist.all_gather(recv_buff_views, recv_buff_views[self.mp_id].clone(),
                                group=group, async_op=False)
            else:
                for i in range(len(recv_buff)):
                    if recv_buff[i].is_floating_point():
                        recv_buff_views = [self._mp_view(recv_buff[i], j) for j in range(mp_size)]
                        dist.all_gather(recv_buff_views, recv_buff_views[self.mp_id].clone(),
                                        group=group, async_op=False)

    def _exec_bps_sync_grads(self, buffer_id):
        name = f'grad_{buffer_id}'
        recv_buff = self.pipe_buffers['bps_grad_recv'][buffer_id]
        if self.mp_id == 0:
            if isinstance(recv_buff, torch.Tensor):
                p2p.bps_sync(self.next_stage, name, index=0)
            else:
                for i in range(len(recv_buff)):
                    p2p.bps_sync(self.next_stage, name, index=i)

        # broadcast the activation at MP rank 0 to other ranks
        if self.grid.model_parallel_size > 1:
            src_rank = self.grid.slice_parallel_src_id
            group = self.grid.slice_proc_group
            if isinstance(recv_buff, torch.Tensor):        
                dist.broadcast(recv_buff, src_rank, group=group, async_op=False)
            else:
                for i in range(len(recv_buff)):
                    if recv_buff[i].is_floating_point():
                        dist.broadcast(recv_buff[i], src_rank, group=group, async_op=False)

    def _exec_bps_sync_partitioned_activations(self, buffer_id):
        recv_buff = self.pipe_buffers['bps_act_recv'][buffer_id]
        recvd = None
        src_rank = self.grid.slice_parallel_src_id
        mp_size = self.grid.model_parallel_size
        group = self.grid.slice_proc_group
        name = f'act_{buffer_id}'

        if isinstance(recv_buff, torch.Tensor):
            p2p.bps_sync(self.prev_stage, name, index=0)
            # broadcast the activation at MP rank 0 to other ranks
            if mp_size > 1:
                recv_buff_views = [self._mp_view(recv_buff, i) for i in range(mp_size)]
                dist.all_gather(recv_buff_views, recv_buff_views[self.mp_id].clone(),
                                group=group, async_op=False)
            recvd = recv_buff.clone().detach()
            recvd.requires_grad = recv_buff.is_floating_point()
        else:
            recvd = [None] * len(recv_buff)
            for i in range(len(recv_buff)):
                p2p.bps_sync(self.prev_stage, name, index=i)
                # broadcast the activation at MP rank 0 to other ranks
                if mp_size > 1:
                    recv_buff_views = [self._mp_view(recv_buff[i], j) for j in range(mp_size)]
                    dist.all_gather(recv_buff_views, recv_buff_views[self.mp_id].clone(),
                                    group=group, async_op=False)
                recvd[i] = recv_buff[i].clone().detach()
            recvd = tuple(recvd)
            for buffer in recvd:
                buffer.requires_grad = buffer.is_floating_point()

        self.pipe_buffers['inputs'][buffer_id] = recvd

    def _exec_bps_sync_activations(self, buffer_id):
        recv_buff = self.pipe_buffers['bps_act_recv'][buffer_id]
        recvd = None
        src_rank = self.grid.slice_parallel_src_id
        group = self.grid.slice_proc_group
        name = f'act_{buffer_id}'

        if isinstance(recv_buff, torch.Tensor):
            if self.mp_id == 0:        
                p2p.bps_sync(self.prev_stage, name, index=0)
            # broadcast the activation at MP rank 0 to other ranks
            if self.grid.model_parallel_size > 1:
                dist.broadcast(recv_buff, src_rank, group=group, async_op=False)
            recvd = recv_buff.clone().detach()
            recvd.requires_grad = recv_buff.is_floating_point()
        else:
            recvd = [None] * len(recv_buff)
            for i in range(len(recv_buff)):
                if self.mp_id == 0:
                    p2p.bps_sync(self.prev_stage, name, index=i)
                # broadcast the activation at MP rank 0 to other ranks
                if self.grid.model_parallel_size > 1:
                    dist.broadcast(recv_buff[i], src_rank, group=group, async_op=False)
                recvd[i] = recv_buff[i].clone().detach()
            recvd = tuple(recvd)
            for buffer in recvd:
                buffer.requires_grad = buffer.is_floating_point()

        self.pipe_buffers['inputs'][buffer_id] = recvd

    def _exec_bps_recv_partitioned_activations(self, buffer_id):
        self._profiling_func_enter('_exec_bps_recv_activations')
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').start()

        recv_buffs = self.pipe_buffers['bps_act_recv']

        # Allocate the buffer if necessary
        if recv_buffs[buffer_id] is None:
            if recv_buffs[0] is None:
                recv_buffs[buffer_id] = self._recv_tensor_meta(self.prev_stage)
            else:
                if torch.is_tensor(recv_buffs[0]):
                    recv_buffs[buffer_id] = recv_buffs[0].clone().detach()
                else:
                    recv_buffs[buffer_id] = tuple([x.clone().detach() for x in recv_buffs[0]])

        assert not self.args.broadcast_activation
        assert not self.is_pipe_partitioned
        recv_buff = recv_buffs[buffer_id]
        name = f'act_{buffer_id}'
        if isinstance(recv_buff, torch.Tensor):
            p2p.bps_recv(self._mp_view(recv_buff, self.mp_id), self.prev_stage,
                         name, index=0, async_op=True)
        else:
            assert isinstance(recv_buff, (tuple, list))
            for idx, buffer in enumerate(recv_buff):
                assert torch.is_tensor(buffer)
                if DS_PIPE_VERBOSE >= 3:
                    log_dist(f'DS BPS_RECV tensors {idx}/{len(recv_buff)}, prev_stage: {self.prev_stage}', ranks=[-1])
                p2p.bps_recv(self._mp_view(buffer, self.mp_id), self.prev_stage,
                             name, index=idx, async_op=True)

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').stop()
        self._profiling_func_exit()

    def _exec_bps_recv_activations(self, buffer_id):
        self._profiling_func_enter('_exec_bps_recv_activations')
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').start()

        recv_buffs = self.pipe_buffers['bps_act_recv']

        # Allocate the buffer if necessary
        if recv_buffs[buffer_id] is None:
            if recv_buffs[0] is None:
                recv_buffs[buffer_id] = self._recv_tensor_meta(self.prev_stage)
            else:
                if torch.is_tensor(recv_buffs[0]):
                    recv_buffs[buffer_id] = recv_buffs[0].clone().detach()
                else:
                    recv_buffs[buffer_id] = tuple([x.clone().detach() for x in recv_buffs[0]])

        assert not self.args.broadcast_activation
        assert not self.is_pipe_partitioned
        recv_buff = recv_buffs[buffer_id]
        if self.mp_id == 0:
            name = f'act_{buffer_id}'
            if isinstance(recv_buff, torch.Tensor):
                p2p.bps_recv(recv_buff, self.prev_stage, name, index=0, async_op=True)
            else:
                assert isinstance(recv_buff, (tuple, list))
                for idx, buffer in enumerate(recv_buff):
                    assert torch.is_tensor(buffer)
                    p2p.bps_recv(buffer, self.prev_stage, name, index=idx, async_op=True)

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').stop()
        self._profiling_func_exit()

    def _exec_bps_recv_partitioned_grads(self, buffer_id):
        self._profiling_func_enter('_exec_bps_recv_grads')
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]
        grad_buffs = self.pipe_buffers['bps_grad_recv']
        # Restore partitioned output if it was partitioned and we are sending full gradients
        assert not self.is_pipe_partitioned
        assert not self.is_grad_partitioned
        assert not self.args.broadcast_grads
        assert ENABLE_BPS_PARTITION
        # Allocate gradient if necessary
        if grad_buffs[buffer_id] is None:
            if isinstance(outputs, torch.Tensor):
                s = list(outputs.size())
                grad_buffs[buffer_id] = self._allocate_buffer(s, num_buffers=1)[0]
            else:
                sizes = [list(t.size()) for t in outputs if t.is_floating_point()]
                grad_buffs[buffer_id] = self._allocate_buffers(sizes, num_buffers=1)[0]
        grad_buff = grad_buffs[buffer_id]
        name = f'grad_{buffer_id}'
        if isinstance(grad_buff, torch.Tensor):
            p2p.bps_recv(self._mp_view(grad_buff, self.mp_id), self.next_stage,
                         name, index=0, async_op=True)
        else:
            assert isinstance(outputs, tuple)
            recv_idx = 0
            for idx, buffer in enumerate(grad_buff):
                p2p.bps_recv(self._mp_view(buffer, self.mp_id), self.next_stage,
                             name, index=recv_idx, async_op=True)
                recv_idx += 1

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').stop()
        self._profiling_func_exit()

    def _exec_bps_recv_grads(self, buffer_id):
        self._profiling_func_enter('_exec_bps_recv_grads')
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]
        grad_buffs = self.pipe_buffers['bps_grad_recv']
        # Restore partitioned output if it was partitioned and we are sending full gradients
        assert not self.is_pipe_partitioned
        assert not self.is_grad_partitioned
        assert not self.args.broadcast_grads
        # Allocate gradient if necessary
        if grad_buffs[buffer_id] is None:
            if isinstance(outputs, torch.Tensor):
                s = list(outputs.size())
                grad_buffs[buffer_id] = self._allocate_buffer(s, num_buffers=1)[0]
            else:
                sizes = [list(t.size()) for t in outputs if t.is_floating_point()]
                grad_buffs[buffer_id] = self._allocate_buffers(sizes, num_buffers=1)[0]
        grad_buff = grad_buffs[buffer_id]
        name = f'grad_{buffer_id}'
        if isinstance(grad_buff, torch.Tensor):
            if self.mp_id == 0:
                p2p.bps_recv(grad_buff, self.next_stage, name, index=0, async_op=True)
        else:
            assert isinstance(outputs, tuple)
            recv_idx = 0
            if self.mp_id == 0:
                for idx, buffer in enumerate(grad_buff):
                    p2p.bps_recv(buffer, self.next_stage, name, index=recv_idx, async_op=True)
                    recv_idx += 1

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').stop()
        self._profiling_func_exit()

    def _exec_optimizer_step(self, lr_kwargs=None):
        self._profiling_func_enter('_exec_optimizer_step')
        if self.wall_clock_breakdown():
            self.timers('step_microstep').start()
            self.timers('step').start()
        self.mem_status('BEFORE STEP', reset_max=True)

        if self.global_rank == 0 and MEGATRON_DEBUG_GRAD:
             params = list(self.module.named_parameters())
             for i in (0, 1, -2, -1):
                 p = params[i]
                 if p[1] is None:
                     print(f'name={p[0]} | None', flush=True)
                 elif p[1].grad is None:
                     print(f'name={p[0]} | weight={p[1].mean()}', flush=True)
                 else:
                     print(f'name={p[0]} | weight={p[1].norm()} | grad={p[1].grad.norm()}', flush=True)
             params_w_grad = []
             params_wo_grad = []
             for p in params:
                 if p[1].grad is not None:
                     params_w_grad.append(p[0])
                 else:
                     params_wo_grad.append(p[0])

        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs)
        self._force_grad_boundary = False

        self.mem_status('AFTER STEP')

        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/lr',
                                        self.get_lr()[0],
                                        self.global_samples)]
                if self.fp16_enabled() and hasattr(self.optimizer, 'cur_scale'):
                    self.summary_events.append((f'Train/Samples/loss_scale',
                                                self.optimizer.cur_scale,
                                                self.global_samples))
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])

        if self.wall_clock_breakdown():
            self.timers('step_microstep').stop()
            self.timers('step').stop()
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'batch_input',
                    'forward_microstep',
                    'backward_microstep',
                    'backward_inner_microstep',
                    'backward_allreduce_microstep',
                    'backward_tied_allreduce_microstep',
                    'step_microstep'
                ])
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'forward',
                    'backward',
                    'backward_inner',
                    'backward_allreduce',
                    'step'
                ])
        self._profiling_func_exit()

    def _zero_grads(self, inputs):
        if isinstance(inputs, torch.Tensor):
            if inputs.grad is not None:
                inputs.grad.data.zero_()
        else:
            for t in inputs:
                if t.grad is not None:
                    t.grad.data.zero_()

    def _allocate_zeros(self, shape, fp16=None, **kwargs):
        """ Allocate a tensor of zeros on the engine's device.

        Arguments:
            shape: the shape of the tensor to allocate
            fp16 (bool): whether to use FP16. default: defer to self.fp16_enabled()
            kwargs: passed to torch.zeros()

        Returns:
            A tensor from torch.zeros() allocated on self.device.
        """

        if fp16 is None:
            fp16 = self.fp16_enabled()

        if fp16:
            return torch.zeros(shape, dtype=torch.half, device=self.device, **kwargs)
        else:
            return torch.zeros(shape, device=self.device, **kwargs)

    def _allocate_zeros2(self, shape, dtype, **kwargs):
        return torch.zeros(shape, dtype=dtype, device=self.device, **kwargs)

    def _allocate_buffer(self, shape, num_buffers=-1, **kwargs):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffers.append(self._allocate_zeros(shape, **kwargs))
        return buffers

    def _allocate_buffer2(self, shape, dtype, num_buffers=-1, **kwargs):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffers.append(self._allocate_zeros2(shape, dtype, **kwargs))
        return buffers

    def _allocate_buffers(self, shapes, requires_grad=False, num_buffers=-1):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffer = []
            for shape in shapes:
                buffer.append(self._allocate_zeros(shape, requires_grad=requires_grad))
            buffers.append(buffer)
        return buffers

    def _allocate_buffers2(self, shapes, dtypes, requires_grad=False, num_buffers=-1):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffer = []
            for i in range(len(shapes)):
                buffer.append(self._allocate_zeros2(shapes[i], dtypes[i], requires_grad=requires_grad))
            buffers.append(buffer)
        return buffers

    def forward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def backward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def step(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.BytePSForwardPass: _exec_bps_forward_pass,
        schedule.BytePSBackwardPass: _exec_bps_backward_pass,
        schedule.BytePSSendActivation: _exec_bps_send_partitioned_activations if ENABLE_BPS_PARTITION else _exec_bps_send_activations,
        schedule.BytePSRecvActivation: _exec_bps_recv_partitioned_activations if ENABLE_BPS_PARTITION else _exec_bps_recv_activations,
        schedule.BytePSSyncActivation: _exec_bps_sync_partitioned_activations if ENABLE_BPS_PARTITION else _exec_bps_sync_activations,
        schedule.BytePSSyncGrad: _exec_bps_sync_partitioned_grads if ENABLE_BPS_PARTITION else _exec_bps_sync_grads,
        schedule.BytePSSendGrad: _exec_bps_send_partitioned_grads if ENABLE_BPS_PARTITION else _exec_bps_send_grads,
        schedule.BytePSRecvGrad: _exec_bps_recv_partitioned_grads if ENABLE_BPS_PARTITION else _exec_bps_recv_grads,
        schedule.BytePSSyncAll: _exec_bps_sync_all
    }

    def _exec_schedule(self, pipe_schedule):
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        self.fwd_outputs = []

        # For each step in the schedule
        has_optim_step = False
        for step_cmds in pipe_schedule:
            # For each instruction in the step
            for cmd in step_cmds:
                if isinstance(cmd, schedule.OptimizerStep):
                    has_optim_step = True
                if DS_PIPE_VERBOSE:
                    if "buffer_id" in cmd.kwargs:
                        print(f'[{self.grid.get_global_rank()}] | cmd={cmd.__class__.__name__} | {cmd.kwargs["buffer_id"]}', flush=True)
                    else:
                        print(f'[{self.grid.get_global_rank()}] | cmd={cmd.__class__.__name__}', flush=True)
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f'{self.__class__.__name__} does not understand instruction {repr(cmd)}'
                    )

                self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                self._exec_instr(**cmd.kwargs)
        # check for anormalies
        if isinstance(pipe_schedule, (schedule.BytePSTrainSchedule, schedule.TrainSchedule)):
            assert has_optim_step

    def broadcast_last_to_first(self, data):
        if self.is_last_stage():
            src_rank = self.global_rank
        else:
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
        
        dist.broadcast(tensor=data,
                        src=src_rank,
                        group=self.grid.get_pipe_parallel_group())

    def broadcast_first_to_last(self, data):
        if self.is_first_stage():
            src_rank = self.global_rank
        else:
            src_rank = self.grid.stage_to_global(0)
        
        dist.broadcast(tensor=data,
                        src=src_rank,
                        group=self.grid.get_pipe_parallel_group())


    def load_megatron_checkpoint(self,
                                 load_dir,
                                 tag=None,
                                 load_module_strict=True,
                                 load_optimizer_states=True,
                                 load_lr_scheduler_states=True,
                                 optimizer=None,
                                 lr_scheduler=None,
                                 num_layers=None):

        load_path = get_ckpt_name(load_dir)
        if not os.path.exists(load_path):
            logger.warn(
                'Client provided checkpoint load path: {} does not exist ... skip checkpoint load'
                .format(load_path))
            return None, None

        log_dist(f'loading checkpoint: {load_path}', ranks=[-1])
        checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
        assert num_layers, num_layers
        load_optimizer = True if optimizer and load_optimizer_states else False

        param_indices = load_megatron_model_state(self.module, num_layers, checkpoint, load_optimizer, load_module_strict)

        if lr_scheduler and load_lr_scheduler_states:
            if 'start_lr' not in checkpoint['lr_scheduler']:
                logger.info(f'rank: {self.global_rank} start_lr not found in the checkpoint. Converting lr schedule from Megatron lr scheduler..')
                checkpoint['lr_scheduler']['start_lr'] = checkpoint['lr_scheduler']['max_lr']
                checkpoint['lr_scheduler']['warmup_iter'] = checkpoint['lr_scheduler']['warmup_steps']
                checkpoint['lr_scheduler']['num_iters'] = checkpoint['iteration']
                checkpoint['lr_scheduler']['last_iter'] = checkpoint['iteration'] - 1
                checkpoint['lr_scheduler']['end_iter'] = checkpoint['lr_scheduler']['num_steps']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        import megatron
        args = megatron.get_args()
        if load_optimizer:
            initial_loss_scale = 1.0 * 2**16
            checkpoint['optimizer']['cur_scale'] = initial_loss_scale
            checkpoint['optimizer']['cur_iter'] = checkpoint['iteration']
            checkpoint['optimizer']['dynamic_loss_scale'] = True
            checkpoint['optimizer']['last_overflow_iter'] = checkpoint['iteration'] - 1
            checkpoint['optimizer']['scale_factor'] = 2.0
            import deepspeed
            assert isinstance(optimizer, deepspeed.runtime.fp16.unfused_optimizer.FP16_UnfusedOptimizer)
            num_param_groups = len(optimizer.optimizer.param_groups)

            # set up the subset of optimizer states that this pipeline stage cares
            ckpt_param_groups = checkpoint['optimizer']['optimizer']['param_groups']
            ckpt_states = checkpoint['optimizer']['optimizer']['state']
            ckpt_fp32_params = checkpoint['optimizer']['fp32_from_fp16_params']
            ckpt_states_subset = type(ckpt_states)()

            # all relevant parameter indices, including the ones with weight_decay, and the ones without wd.
            param_index_begin = 0
            pp_rank = self.mpu.get_pipe_parallel_rank()
            cross_group_param_size_list = []
            with torch.no_grad():
                if self.is_last_stage():
                    ckpt_states_subset[0] = ckpt_states[0]

                for group_idx in range(num_param_groups):
                    contains_shared_embed = group_idx == 0 and self.is_last_stage()
                    # param_size_tensor does not include the shared embed tensor
                    param_size_tensor = torch.zeros(args.num_stages, dtype=torch.int32).cuda()
                    param_size_tensor[pp_rank] = len(param_indices[group_idx])
                    if contains_shared_embed:
                        param_size_tensor[pp_rank] -= 1
                    # get a global view of all tensor count
                    torch.distributed.all_reduce(param_size_tensor, group=self.mpu.get_pipe_parallel_group())
                    param_size_list = param_size_tensor.cpu().numpy().tolist()
                    cross_group_param_size_list.extend(param_size_list)
                    accumulate_param_size_list = []
                    accumulate_param_size_list.append(0)
                    for param_size in param_size_list:
                        accumulate_param_size_list.append(accumulate_param_size_list[-1] + param_size)
                    total_size = param_size_tensor.sum().cpu().numpy()
                    ckpt_params_size = len(ckpt_param_groups[group_idx]["params"])
                    assert total_size == ckpt_params_size, (group_idx, total_size, ckpt_params_size)
                    ckpt_fp32_subset = [ckpt_fp32_params[group_idx][0]] if contains_shared_embed else []
                    start_idx = accumulate_param_size_list[pp_rank]
                    end_idx = accumulate_param_size_list[pp_rank + 1]
                    ckpt_fp32_subset.extend(ckpt_fp32_params[group_idx][start_idx:end_idx])
                    """ select fp32 master weights subset from the Megatron checkpoint

                    len(fp32_from_fp16_params): 2, type(fp32_from_fp16_params): list

                    fp32_from_fp16_params is organized in two param_groups:
                    - fp32_from_fp16_params[0] is for params with wd, with length = 122
                    - fp32_from_fp16_params[1] is for params without wd, with length = 242

                    """
                    checkpoint['optimizer']['fp32_from_fp16_params'][group_idx] = ckpt_fp32_subset
                    ckpt_param_groups[group_idx]["params"] = list(range(param_index_begin, param_index_begin + len(ckpt_fp32_subset)))
                    param_index_begin += len(ckpt_fp32_subset)

                """
                select state subset from the Megatron checkpoint.

                the Megatron checkpoint 'state' is organized in the following way:

                len(state) = 364

                - key = 0,   val = state of group 0 param 0
                - key = 1,   val = state of group 0 param 1
                - key = 2,   val = state of group 0 param 2
                ...
                - key = 121, val = state of group 0 param 121
                - key = 122, val = state of group 1 param 0
                - key = 123, val = state of group 1 param 1
                ...
                - key = 363, val = state of group 1 param 241

                """
                state_idx = 0
                for index, param_size in enumerate(cross_group_param_size_list):
                    start_idx = state_idx
                    end_idx = state_idx + param_size
                    if index % args.num_stages == pp_rank:
                        for key in range(start_idx, end_idx):
                            ckpt_states_subset[len(ckpt_states_subset)] = ckpt_states[key]
                    state_idx = end_idx

            checkpoint['optimizer']['fp32_groups'] = checkpoint['optimizer']['fp32_from_fp16_params']
            checkpoint['optimizer']['optimizer']['state'] = ckpt_states_subset
            checkpoint['optimizer']['optimizer_state_dict'] = checkpoint['optimizer']['optimizer']

            # other misc optimizer information
            checkpoint['optimizer']['scale_window'] = args.loss_scale_window
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info(f'rank: {self.global_rank} loaded optimizer state from Megatron checkpoint.')

        # random states
        import random, numpy as np
        self.global_steps = checkpoint['iteration']
        random.setstate(checkpoint['random_rng_state'])
        np.random.set_state(checkpoint['np_rng_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        megatron.mpu.get_cuda_rng_tracker().set_states(checkpoint['rng_tracker_states'])
        logger.info(f'rank: {self.global_rank} loaded randomness state from Megatron checkpoint. DONE loading all states')
        return self.global_steps


    def _get_ckpt_name(self, checkpoints_path, tag):
        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        pp_rank = 0 if self.mpu is None else self.mpu.get_pipe_parallel_rank()
        if self.zero_optimization_partition_weights():
            filename = 'zero_pp_rank_{}'.format(
                torch.distributed.get_rank(group=self.optimizer.dp_process_group))
            ckpt_name = os.path.join(
                checkpoints_path,
                str(tag),
                filename + '_mp_rank_{:02d}_pp_rank_{:02d}'.format(mp_rank, pp_rank) + '_model_states.pt')
        else:
            ckpt_name = os.path.join(
                checkpoints_path,
                str(tag),
                'mp_rank_{:02d}_pp_rank_{:02d}'.format(mp_rank, pp_rank) + '_model_states.pt')
        return ckpt_name
