# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from megatron import get_args
from megatron import mpu

# try:
#     import veGiantModel
# except ImportError:
#     byteGiantModel = None

class MockModule(nn.Module):
    """Module for testing model parallelism"""
    pass

try:
    from th_fastertransformer import Linear

    class LinearFunction(autograd.Function):

        @staticmethod
        def forward(ctx, input_tensor, weight, bias, act_gelu=False, dropout_rate=0.0):
            bias_out = torch.Tensor(0)
            dropout_mask = torch.Tensor(0)
            if act_gelu == True or dropout_rate > 0.0:
                output, bias_out, dropout_mask = Linear.forward_gelu_dropout(input_tensor, weight, bias, act_gelu, dropout_rate)
            else:
                output = Linear.forward(input_tensor, weight, bias)
            ctx.save_for_backward(input_tensor, weight, bias_out, dropout_mask)
            ctx.act_gelu = act_gelu
            ctx.dropout_rate = dropout_rate
            return output

        @staticmethod
        def backward(ctx, grad_out):
            act_gelu = ctx.act_gelu
            dropout_rate = ctx.dropout_rate
            input_tensor, weight, bias_out, dropout_mask = ctx.saved_tensors
            if act_gelu == True or dropout_rate > 0.0:
                grad_in, grad_weight, grad_bias = Linear.backward_gelu_dropout(
                    grad_out, input_tensor, weight, act_gelu, dropout_rate, bias_out, dropout_mask)
            else:
                grad_in, grad_weight, grad_bias = Linear.backward(
                    grad_out, input_tensor, weight)
            return grad_in, grad_weight, grad_bias, None, None

    class FTLinear(nn.Module):
        def __init__(self, in_features, out_features, initializer_range=0.02, act_gelu=False, dropout_rate=0.0):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.act_gelu = act_gelu
            self.dropout_rate = dropout_rate

            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias.data.zero_()

        def forward(self, input_tensor):
            return LinearFunction.apply(input_tensor, self.weight, self.bias, self.act_gelu, self.dropout_rate if self.training else 0.)

        def extra_repr(self):
            return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

except Exception as e:
    FTLinear = None

try:
    from th_fastertransformer import LinearTranspose

    class LinearTransposeFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight, bias, head_num, transpose_type):
            output = LinearTranspose.forward(input_tensor, weight, bias, head_num, transpose_type)
            ctx.head_num = head_num
            ctx.transpose_type = transpose_type
            ctx.save_for_backward(input_tensor, weight)
            return output

        @staticmethod
        def backward(ctx, grad_out):
            input_tensor, weight = ctx.saved_tensors
            grad_in, grad_weight, grad_bias = LinearTranspose.backward(grad_out, input_tensor, weight, ctx.head_num, ctx.transpose_type)
            return grad_in, grad_weight, grad_bias, None, None

    class FTLinearTranspose(nn.Module):
        def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.head_num = head_num
            self.transpose_type = transpose_type
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias = nn.Parameter(torch.Tensor(out_features))

            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias.data.zero_()

        def forward(self, input_tensor):
            return LinearTransposeFunction.apply(input_tensor, self.weight, self.bias, self.head_num, self.transpose_type)

        def extra_repr(self):
            return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

except Exception as e:
    FTLinearTranspose = None
    FTDAGather = None

def column_parallel_load_hook(module, log_fn):
    """hook for column parallel linear's load_state_dict function.
    It is a helper function to load a the checkpoint from a
    non-model-parallel module. It returns a hook function that
    pre-processes the checkpoint to parallel slices such that
    each model parallel rank could load the corresponding slice.

    Arguments:
        module: ColumnParallelLinear or ColumnParallelLinearTranspose

        log_fn: function for logging

    Returns:
        A hook function to help load model parallel modules from non-
        model-parallel checkpoints.
    """
    assert module.mp_rank is not None
    assert module.out_features is not None
    def hook(state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs):
        weight_name = prefix + 'weight'
        bias_name = prefix + 'bias'
        if weight_name in state_dict:
            v = state_dict[weight_name]
            assert len(v.shape) == 2, v.shape
            idx_begin = module.mp_rank * module.out_features
            idx_end = (module.mp_rank + 1) * module.out_features
            shard = v[idx_begin:idx_end, :]
            state_dict[weight_name] = shard
            log_fn(f"slice param {weight_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
        if bias_name in state_dict:
            v = state_dict[bias_name]
            assert len(v.shape) == 1, v.shape
            idx_begin = module.mp_rank * module.out_features
            idx_end = (module.mp_rank + 1) * module.out_features
            shard = v[idx_begin:idx_end]
            state_dict[bias_name] = shard
            log_fn(f"slice param {bias_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
    return hook

def column_serial_load_hook(module, log_fn):
    """hook for column serial linear's load_state_dict function.
    It is a helper function to load a the checkpoint from a
    non-model-parallel module. It returns a hook function that
    pre-processes the checkpoint to parallel slices such that
    each model parallel rank could load the corresponding slice.

    Arguments:
        module: ColumnSerialLinear or ColumnSerialLinearTranspose

        log_fn: function for logging

    Returns:
        A hook function to help load model serial modules from non-
        model-parallel checkpoints.
    """
    assert module.model_parallel_size is not None
    assert module.out_features is not None
    def hook(state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs):
        weight_name = prefix + 'weight'
        bias_name = prefix + 'bias'
        if weight_name in state_dict:
            v = state_dict[weight_name]
            assert len(v.shape) == 2, v.shape
            for i in range(module.model_parallel_size):
                weight_name_i = weight_name + "." + str(i)
                idx_begin = i * module.out_features
                idx_end = (i + 1) * module.out_features
                shard = v[idx_begin:idx_end, :]
                state_dict[weight_name_i] = shard
                log_fn(f"slice param {weight_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
            del state_dict[weight_name]
        if bias_name in state_dict:
            v = state_dict[bias_name]
            assert len(v.shape) == 1, v.shape
            for i in range(module.model_parallel_size):
                bias_name_i = bias_name + "." + str(i)
                idx_begin = i * module.out_features
                idx_end = (i + 1) * module.out_features
                shard = v[idx_begin:idx_end]
                state_dict[bias_name_i] = shard
                log_fn(f"slice param {bias_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
            del state_dict[bias_name]
    return hook

class ColumnSerialLinear(MockModule):
    def __init__(self, in_features, out_features, initializer_range=0.02,
                 act_gelu=False, dropout_rate=0.0, load_from_shards=False, use_ft=False):
        """
        A serial module that mocks the ColumnParallelLinear module. It mocks the parallel
        logic by applying the series of work on the same rank, and reduce the result if needed.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.in_features = in_features
        self.out_features = out_features // model_parallel_size
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        bias_params = [nn.Parameter(torch.Tensor(self.out_features)) for _ in range(model_parallel_size)]
        self.bias = nn.ParameterList(bias_params)
        self.act_gelu = act_gelu
        self.dropout_rate = dropout_rate
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        for bias in self.bias:
            bias.data.zero_()
        self.use_ft = use_ft
        if not use_ft:
            assert not act_gelu
            assert not dropout_rate, dropout_rate
        if not load_from_shards:
            load_hook = column_serial_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearFunction.apply(input_tensor, self.weight[i], self.bias[i], self.act_gelu,
                                                self.dropout_rate if self.training else 0.)
            else:
                output_i = nn.functional.linear(input_tensor, self.weight[i], self.bias[i])
            outputs.append(output_i)
        output = torch.cat(outputs, dim=-1)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, initializer_range=0.02,
                 act_gelu=False, dropout_rate=0.0, load_from_shards=False, use_ft=False,
                 bias=True, gather_output=False):
        """Linear layer with column parallelism.

        The linear layer is defined as Y = dropout(gelu(XA + b)). A is parallelized along
        its second dimension as A = [A_1, ..., A_p].

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            act_gelu: If true, apply gelu activation to (XA+b)
            dropout_rate: If greater than zero, apply dropout to gelu(XA+b)
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
            use_ft: use faster transformer for acceleration.
            bias: If true, add bias
            gather_output: If true, call all-gether on output and make Y avaiable
                        to all GPUs, otherwise, every GPU will have its output
                        which is Y_i = XA_i
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.in_features = in_features
        self.out_features = out_features // model_parallel_size
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            self.bias.data.zero_()
        else:
            self.bias = None
            assert not use_ft
        self.gather_output = gather_output
        self.act_gelu = act_gelu
        self.dropout_rate = dropout_rate
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not use_ft:
            assert not act_gelu
            assert not dropout_rate, dropout_rate
        if not load_from_shards:
            load_hook = column_parallel_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        import veGiantModel
        input_tensor = veGiantModel.distributed.copy_to_model_parallel_region(input_tensor)
        if self.use_ft:
            output = LinearFunction.apply(input_tensor, self.weight, self.bias, self.act_gelu,
                                            self.dropout_rate if self.training else 0.)
        else:
            output = nn.functional.linear(input_tensor, self.weight, self.bias)
        if self.gather_output:
            output = veGiantModel.distributed.gather_from_model_parallel_region(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class RowSerialLinear(MockModule):
    def __init__(self, in_features, out_features, initializer_range=0.02, dropout_rate=0.0,
                 load_from_shards=False, use_ft=False):
        """
        A serial module that mocks the RowParallelLinear module. It mocks the parallel
        logic by applying the series of work on the same rank.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.in_features = in_features // model_parallel_size
        self.out_features = out_features
        assert in_features % model_parallel_size == 0, (in_features, model_parallel_size)
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.dropout_rate = dropout_rate
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not load_from_shards:
            def load_hook(state_dict, prefix, local_metadata, strict, missing_keys,
                          unexpected_keys, error_msgs):
                weight_name = prefix + 'weight'
                if weight_name in state_dict:
                    v = state_dict[weight_name]
                    assert len(v.shape) == 2, v.shape
                    for i in range(model_parallel_size):
                        weight_name_i = weight_name + '.' + str(i)
                        idx_begin = i * self.in_features
                        idx_end = (i + 1) * self.in_features
                        shard = v[:, idx_begin:idx_end]
                        state_dict[weight_name_i] = shard
                        print(f"slice param {weight_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
                    del state_dict[weight_name]
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        input_tensors = torch.split(input_tensor, self.in_features, dim=-1)
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearFunction.apply(input_tensors[i].contiguous(), self.weight[i], self.bias, False, 0.)
            else:
                output_i = nn.functional.linear(input_tensors[i].contiguous(), self.weight[i], self.bias)
            outputs.append(output_i)
        output = outputs[0]
        for i in range(self.model_parallel_size - 1):
            output = output + outputs[i + 1]
        if self.dropout_rate:
            output = self.dropout(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, initializer_range=0.02, dropout_rate=0.0,
                 load_from_shards=False, use_ft=False):
        """Linear layer with row parallelism.

        The linear layer is defined as Y = XA + b. A is parallelized along
        its first dimension and X along its second dimension as:
                -   -
                | A_1 |
                | .   |
            A = | .   |        X = [X_1, ..., X_p]
                | .   |
                | A_p |
                -   -

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            dropout_rate: If greater than zero, apply dropout XA+b
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
            use_ft: use faster transformer for acceleration.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.in_features = in_features // model_parallel_size
        self.out_features = out_features
        assert in_features % model_parallel_size == 0, (in_features, model_parallel_size)
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.dropout_rate = dropout_rate
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not load_from_shards:
            def load_hook(state_dict, prefix, local_metadata, strict, missing_keys,
                            unexpected_keys, error_msgs):
                weight_name = prefix + 'weight'
                if weight_name in state_dict:
                    v = state_dict[weight_name]
                    assert len(v.shape) == 2, v.shape
                    idx_begin = self.mp_rank * self.in_features
                    idx_end = (self.mp_rank + 1) * self.in_features
                    shard = v[:, idx_begin:idx_end]
                    state_dict[weight_name] = shard
                    print(f"slice param {weight_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        if self.use_ft:
            output = LinearFunction.apply(input_tensor, self.weight, self.bias, False, 0.)
        else:
            output = nn.functional.linear(input_tensor, self.weight, self.bias)
        import veGiantModel
        output = veGiantModel.distributed.reduce_from_model_parallel_region(output)

        if self.dropout_rate:
            output = self.dropout(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class ColumnParallelLinearTranspose(nn.Module):
    def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02,
                 use_ft=False, load_from_shards=False):
        """Linear layer with column parallelism. The output is then reshaped to 4D with
        (dim0, dim1, head_num, out_features / head_num), then permuted with axies provided by transpose_type.
        For equivalent computation, check the implementation of `ColumnSerialLinearTranspose`.

        The linear layer is defined as Y = XA + b. A is parallelized along
        its second dimension as A = [A_1, ..., A_p].

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            head_num: number of "heads" for the out_feature dimension.
            transpose_type: the axies for permutation on the output.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            use_ft: use faster transformer for acceleration.
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
        """
        super().__init__()
        self.use_ft = use_ft
        self.in_features = in_features
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()

        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.out_features = out_features // model_parallel_size
        assert head_num % model_parallel_size == 0, (head_num, model_parallel_size)
        self.head_num = head_num // model_parallel_size
        self.head_dim = self.out_features // self.head_num
        self.transpose_type = transpose_type
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        if not load_from_shards:
            load_hook = column_parallel_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        import veGiantModel
        input_tensor = veGiantModel.distributed.copy_to_model_parallel_region(input_tensor)
        if self.use_ft:
            output = LinearTransposeFunction.apply(input_tensor, self.weight, self.bias,
                                                    self.head_num, self.transpose_type)
        else:
            assert self.transpose_type == "0213", self.transpose_type
            linear_out = nn.functional.linear(input_tensor, self.weight, self.bias)
            new_shape = linear_out.size()[:-1] + (self.head_num, self.head_dim)
            linear_out = linear_out.view(*new_shape)
            output = linear_out.permute(0, 2, 1, 3).contiguous()
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

class ColumnSerialLinearTranspose(MockModule):
    def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02,
                    use_ft=False, load_from_shards=False):
        """
        A serial module that mocks the ColumnParallelLinearTranspose module. It mocks the parallel
        logic by applying the series of work on the same rank.
        """
        super().__init__()
        self.use_ft = use_ft
        self.in_features = in_features
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.out_features = out_features // model_parallel_size
        assert head_num % model_parallel_size == 0, (head_num, model_parallel_size)
        self.head_num = head_num // model_parallel_size
        self.head_dim = self.out_features // self.head_num
        self.transpose_type = transpose_type
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        bias_params = [nn.Parameter(torch.Tensor(self.out_features)) for _ in range(model_parallel_size)]
        self.bias = nn.ParameterList(bias_params)
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        for bias in self.bias:
            bias.data.zero_()

        if not load_from_shards:
            load_hook = column_serial_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearTransposeFunction.apply(input_tensor, self.weight[i], self.bias[i], self.head_num, self.transpose_type)
            else:
                assert self.transpose_type == "0213", self.transpose_type
                linear_out = nn.functional.linear(input_tensor, self.weight[i], self.bias[i])
                new_shape = linear_out.size()[:-1] + (self.head_num, self.head_dim)
                linear_out = linear_out.view(*new_shape)
                output_i = linear_out.permute(0, 2, 1, 3).contiguous()
            outputs.append(output_i)
        output = torch.cat(outputs, dim=1)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size,
                                                  rank, world_size):
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f, index_l

def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
    per_partition_vocab_size = mpu.divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size, rank, world_size)

class _VocabParallelCrossEntropy(autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):
        import veGiantModel
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=veGiantModel.distributed.get_model_parallel_group())
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indecies
        get_vocab_range = vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = veGiantModel.distributed.get_model_parallel_rank()
        world_size = veGiantModel.distributed.get_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                 device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(predicted_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=veGiantModel.distributed.get_model_parallel_group())

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=veGiantModel.distributed.get_model_parallel_group())

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= (
            1.0 - target_mask.view(-1).float())

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target):
    """Helper function for the cross entropy."""
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)

def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride
    
    with mpu.get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False):
    import veGiantModel
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(dtype=args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = mpu.divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = veGiantModel.distributed.get_model_parallel_rank()
    world_size = veGiantModel.distributed.get_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None

class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        import veGiantModel
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            vocab_range_from_global_vocab_size(
                self.num_embeddings, veGiantModel.distributed.get_model_parallel_rank(),
                self.model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=args.params_dtype))
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)

    def forward(self, input_):
        if self.model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = mpu.reduce_from_model_parallel_region(output_parallel)
        return output