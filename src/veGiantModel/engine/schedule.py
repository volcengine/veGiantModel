# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
from deepspeed.runtime.pipe.schedule import (
    BufferOpInstruction,PipeInstruction,
    ReduceTiedGrads,ReduceGrads,OptimizerStep,
    LoadMicroBatch,PipeSchedule,TrainSchedule,
)

import os

BYTEPS_REDUCED_MEM = os.environ.get('BYTEPS_REDUCED_MEM', '1') != '0'

class BytePSInferenceSchedule(PipeSchedule):
    """A schedule for inferencing batches using pipeline parallelism.
    """
    def __init__(self, micro_batches, stages, stage_id, prefetch=True):
        super().__init__(micro_batches, stages, stage_id)
        self.prefetch = prefetch

    def steps(self):
        """"""
        total_steps = self.micro_batches + self.stages - 1
        for step_id in range(total_steps):
            cmds = []
            micro_batch_id = step_id - self.stage_id

            buffer_id = micro_batch_id % self.num_pipe_buffers()
            batch_is_valid = self._valid_micro_batch(micro_batch_id)

            if not self.prefetch:    
                if batch_is_valid:
                    if self.is_first_stage or self.is_last_stage:
                        cmds.append(LoadMicroBatch(buffer_id))
                    if self._valid_stage(self.prev_stage):
                        cmds.append(BytePSRecvActivation(buffer_id))
                        cmds.append(BytePSSyncActivation(buffer_id))
                    cmds.append(BytePSForwardPass(buffer_id))
                    if self._valid_stage(self.next_stage):
                        cmds.append(BytePSSendActivation(buffer_id))
            else:
                next_buffer_id = (micro_batch_id + 1) % self.num_pipe_buffers()
                next_batch_is_valid = self._valid_micro_batch(micro_batch_id + 1)
                # micro_batch starts at 0. Get the current batch, and start prefetching
                if micro_batch_id == 0:
                    if self.is_first_stage or self.is_last_stage:
                        cmds.append(LoadMicroBatch(buffer_id))
                    if self._valid_stage(self.prev_stage):
                        cmds.append(BytePSRecvActivation(buffer_id))
                        if next_batch_is_valid:
                            cmds.append(BytePSRecvActivation(next_buffer_id))
                        cmds.append(BytePSSyncActivation(buffer_id))
                    cmds.append(BytePSForwardPass(buffer_id))
                    if self._valid_stage(self.next_stage):
                        cmds.append(BytePSSendActivation(buffer_id))
                elif batch_is_valid:
                    # After micro_batch 0, we prefetch the next one,
                    # and wait for the current one
                    if self._valid_stage(self.prev_stage) and next_batch_is_valid:
                        cmds.append(BytePSRecvActivation(next_buffer_id))
                    if self.is_first_stage or self.is_last_stage:
                        cmds.append(LoadMicroBatch(buffer_id))
                    if self._valid_stage(self.prev_stage):
                        cmds.append(BytePSSyncActivation(buffer_id))
                    cmds.append(BytePSForwardPass(buffer_id))
                    if self._valid_stage(self.next_stage):
                        cmds.append(BytePSSendActivation(buffer_id))

            yield cmds

    def num_pipe_buffers(self):
        """Only `self.micro_batches` pipeline buffers are required for inferencing.

        Returns:
            ``self.micro_batches``
        """
        buffers = min(self.micro_batches, self.stages * 2)
        if BYTEPS_REDUCED_MEM:
            buffers = min(self.stages + 1, self.micro_batches)
        return max(2, buffers)


class BytePSTrainSchedule(TrainSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """
    def __init__(self, micro_batches, stages, stage_id, prefetch=True):
        super().__init__(micro_batches, stages, stage_id)
        self.prefetch = prefetch and micro_batches > 1
        if not self.prefetch:
            print('BYTEPS NO PREFETCH STEPS', flush=True)

    def steps(self):
        if self.prefetch:
            return self._steps()
        else:
            return self._steps_no_prefetch()

    def _steps(self):
        """"""
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            cmds = []
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)
            batch_is_valid = self._valid_micro_batch(micro_batch_id)
            if not batch_is_valid:
                if step_id == total_steps - 1:
                    cmds.append(BytePSSyncAll())
                    cmds.append(ReduceTiedGrads())
                    cmds.append(ReduceGrads())
                    cmds.append(OptimizerStep())
                    yield cmds
                continue
            curr_buffer = self._buffer_idx(micro_batch_id)

            # try to find the next valid batch
            next_step_id = step_id + 1
            next_micro_batch_id, next_is_forward, next_batch_is_valid = None, None, None
            while next_step_id < total_steps:
                next_micro_batch_id, next_is_forward = self._step_to_micro_batch(next_step_id)
                next_batch_is_valid = self._valid_micro_batch(next_micro_batch_id)
                if next_batch_is_valid:
                    break
                next_step_id += 1

            next_buffer = None
            if next_batch_is_valid:
                next_buffer = self._buffer_idx(next_micro_batch_id)

            if micro_batch_id == 0 and is_forward:
                # first/last stage loads
                if self.stage_id == 0 or self.stage_id == self.stages - 1:
                    cmds.append(LoadMicroBatch(curr_buffer))
                # fetch
                if self._valid_stage(self.prev_stage):
                    cmds.append(BytePSRecvActivation(curr_buffer))
                # pre-fetch
                if next_batch_is_valid:
                    if self._valid_stage(self.prev_stage) and next_is_forward:
                        cmds.append(BytePSRecvActivation(next_buffer))
                    if self._valid_stage(self.next_stage) and not next_is_forward:
                        cmds.append(BytePSRecvGrad(next_buffer))
                # sync and compute
                if self._valid_stage(self.prev_stage):
                    cmds.append(BytePSSyncActivation(curr_buffer))
                cmds.append(BytePSForwardPass(curr_buffer))
                if self._valid_stage(self.next_stage):
                    cmds.append(BytePSSendActivation(curr_buffer))
            else:
                # prefetch
                if next_batch_is_valid:
                    if self._valid_stage(self.prev_stage) and next_is_forward:
                        cmds.append(BytePSRecvActivation(next_buffer))
                    if self._valid_stage(self.next_stage) and not next_is_forward:
                        cmds.append(BytePSRecvGrad(next_buffer))
                if is_forward:
                    if self.stage_id == 0 or self.stage_id == self.stages - 1:
                        # First/last stage loads
                        cmds.append(LoadMicroBatch(curr_buffer))
                    if self._valid_stage(self.prev_stage):
                        cmds.append(BytePSSyncActivation(curr_buffer))
                    cmds.append(BytePSForwardPass(curr_buffer))
                    if self._valid_stage(self.next_stage):
                        cmds.append(BytePSSendActivation(curr_buffer))
                else:
                    if self._valid_stage(self.next_stage):
                        cmds.append(BytePSSyncGrad(curr_buffer))
                    cmds.append(BytePSBackwardPass(curr_buffer))
                    if self._valid_stage(self.prev_stage):
                        cmds.append(BytePSSendGrad(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(BytePSSyncAll())
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            yield cmds

    def _steps_no_prefetch(self):
        """"""
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            cmds = []
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)
            batch_is_valid = self._valid_micro_batch(micro_batch_id)
            if not batch_is_valid:
                if step_id == total_steps - 1:
                    cmds.append(BytePSSyncAll())
                    cmds.append(ReduceTiedGrads())
                    cmds.append(ReduceGrads())
                    cmds.append(OptimizerStep())
                    yield cmds
                continue

            curr_buffer = self._buffer_idx(micro_batch_id)

            if is_forward:
                if self._valid_stage(self.prev_stage):
                    cmds.append(BytePSRecvActivation(curr_buffer))
                    cmds.append(BytePSSyncActivation(curr_buffer))
                if self.stage_id == 0 or self.stage_id == self.stages - 1:
                    # First/last stage loads
                    cmds.append(LoadMicroBatch(curr_buffer))
                cmds.append(BytePSForwardPass(curr_buffer))
                if self._valid_stage(self.next_stage):
                    cmds.append(BytePSSendActivation(curr_buffer))
            else:
                if self._valid_stage(self.next_stage):
                    cmds.append(BytePSRecvGrad(curr_buffer))
                    cmds.append(BytePSSyncGrad(curr_buffer))
                cmds.append(BytePSBackwardPass(curr_buffer))
                if self._valid_stage(self.prev_stage):
                    cmds.append(BytePSSendGrad(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(BytePSSyncAll())
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            yield cmds

    def num_pipe_buffers(self):
        """As many buffers as the distance from this stage to the last stage.
        """
        buffers = min(self.micro_batches, self.stages * 2)
        if BYTEPS_REDUCED_MEM:
            buffers = min(self.stages + 1, self.micro_batches)
        return max(2, buffers)


class BytePSSendActivation(BufferOpInstruction):
    pass

class BytePSRecvActivation(BufferOpInstruction):
    pass

class BytePSSyncActivation(BufferOpInstruction):
    pass

class BytePSSyncGrad(BufferOpInstruction):
    pass

class BytePSSendGrad(BufferOpInstruction):
    pass

class BytePSRecvGrad(BufferOpInstruction):
    pass

class BytePSForwardPass(BufferOpInstruction):
    pass

class BytePSBackwardPass(BufferOpInstruction):
    pass

class BytePSSyncAll(PipeInstruction):
    pass







