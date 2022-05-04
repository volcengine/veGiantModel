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
from deepspeed.pipe import LayerSpec, TiedLayerSpec
from veGiantModel.engine.module import veGiantModule
from veGiantModel.megatron import get_args, mpu
from veGiantModel.megatron.model.enums import AttnMaskType
from veGiantModel.megatron.model.language_model import (EmbeddingPipe,
                                                        parallel_lm_logits)
from veGiantModel.megatron.model.module import (float16_to_fp32,
                                                fp32_to_float16)
from veGiantModel.megatron.model.transformer import (
    LayerNorm, ParallelTransformerLayerPipe)
from veGiantModel.megatron.model.utils import (init_method_normal,
                                               scaled_init_method_normal)


def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


class GPTModelPipe(veGiantModule):
    """GPT-2 Language model."""

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True):
        args = get_args()
        self.parallel_output = parallel_output

        init_method = init_method_normal(args.init_method_std)

        self.specs = []

        def _to_float16(inputs):
            if args.fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif args.bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        self.specs.append(_to_float16)

        # Embedding layer
        self.specs.append(TiedLayerSpec('embed',
                                        EmbeddingPipe,
                                        args.hidden_size,
                                        args.padded_vocab_size,
                                        args.max_position_embeddings,
                                        args.hidden_dropout,
                                        init_method=init_method,
                                        num_tokentypes=num_tokentypes,
                                        tied_weight_attr='word_embeddings_weight'))

        if args.fp32_residual_connection:
            self.specs.append(lambda x: x.transpose(0, 1).contiguous().float())
        else:
            self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        for layer_idx in range(args.num_layers):
            self.specs.append(
                LayerSpec(ParallelTransformerLayerPipe,
                          init_method=init_method,
                          output_layer_init_method=scaled_init_method_normal(args.init_method_std,
                                                                             args.num_layers),
                          layer_number=layer_idx,
                          self_attn_mask_type=AttnMaskType.causal))

        # Undo data format change
        self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        # Final layernorm after transformer layers
        self.specs.append(
            LayerSpec(LayerNorm,
                      args.hidden_size,
                      eps=args.layernorm_epsilon))

        def _logits_helper(embedding, lm_output):
            """A wrapper to massage inputs/outputs from pipeline. """
            return parallel_lm_logits(
                lm_output,
                embedding.word_embeddings_weight,
                self.parallel_output)

        self.specs.append(
            TiedLayerSpec('embed',
                          EmbeddingPipe,
                          args.hidden_size,
                          args.padded_vocab_size,
                          args.max_position_embeddings,
                          args.hidden_dropout,
                          init_method=init_method,
                          num_tokentypes=num_tokentypes,
                          forward_fn=_logits_helper,
                          tied_weight_attr='word_embeddings_weight')
        )

        # Convert to fp32 if needed
        if args.fp16 or args.bf16:
            self.specs.append(float16_to_fp32)

        # if args.checkpoint_activations:
        #     interval = args.checkpoint_num_layers
        # else:
        interval = 0

        from veGiantModel.launcher.launch import launch_bps
        import os

        rank = int(os.getenv('RANK', '0'))
        device_count = torch.cuda.device_count()
        local_rank = rank % device_count

        if mpu.get_pipeline_model_parallel_world_size() > 1:
            import byteps.torch as bps
            assert bps is not None
            launch_bps(local_rank)

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(num_pp=mpu.get_pipeline_model_parallel_world_size(),
                                             num_mp=mpu.get_tensor_model_parallel_world_size(),
                                             num_dp=mpu.get_data_parallel_world_size())

        super().__init__(layers=self.specs,
                         loss_fn=CrossEntropy,
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method='type:transformer')
        #  topology=topo,
        #  partition_method='type:transformer')
