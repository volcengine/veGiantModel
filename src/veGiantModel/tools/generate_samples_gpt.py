# coding=utf-8
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

"""Sample Generate GPT"""

import torch
import os
import numpy as np
import time
import sys
import json

_cwd = os.path.dirname(os.path.abspath(__file__))
_giantModel_dir = os.path.join(_cwd, '../../')
sys.path.append(_giantModel_dir)

from veGiantModel.model.gpt_piped import GPTModelPiped
from megatron import print_rank_0
from megatron import get_args
from util.text_generation_utils import generate_samples_input_from_file
from deepspeed.utils import log_dist

import veGiantModel

def model_provider():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPTModelPiped(forward_parallel_output=False)
    return model


def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--sample-input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--sample-output-file", type=str, default=None,
                       help='Output file got from --sample-input-file')
    group.add_argument("--num-samples", type=int, default=0,
                       help='Number of samples to generate unconditionally, '
                       'defaults to 0 and interactive conditional sampling')
    group.add_argument("--genfile", type=str,
                       help='Output file when generating unconditionally')
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                       'instead of using previously computed keys/values.')

    return parser

def main():
    """Main program."""
    veGiantModel.initialize_megatron(extra_args_provider=add_text_generate_args)
    args = get_args()

    args.iteration = 0
    model = model_provider()
    _param_dict = json.loads(args.config_param)
    engine, _, _, _ = veGiantModel.initialize(
        model=model,
        args=args,
        mpu=None,
        dist_init_required=False,
        config_params = _param_dict,
    )
    engine.set_batch_fn(model.batch_fn)

    if args.load_megatron is not None:
        log_dist(f'load checkpoint start {args.load}', ranks=[-1])
        engine.load_megatron_checkpoint(args.load_megatron, 
                                        load_optimizer_states=False, 
                                        load_lr_scheduler_states=False, 
                                        num_layers=args.num_layers)
        log_dist(f'load checkpoint finished', ranks=[-1])
    elif args.load is not None:
        log_dist(f'load checkpoint start {args.load}', ranks=[-1])
        engine.load_checkpoint(args.load, 
                                load_optimizer_states=False, 
                                load_lr_scheduler_states=False)

    log_dist(f'start evaluating', ranks=[-1])

    generate_samples_input_from_file(engine)


if __name__ == "__main__":
    main()