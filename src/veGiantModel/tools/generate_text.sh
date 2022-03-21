#!/bin/bash

set -x
date
CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt


export NCCL_DEBUG=WARN
export WORKER_0_HOST=
export WORKER_0_PORT=
export NUM_WORKER=
export WORKER_RANK=
export GPU_PER_WORKER=

GPUS_PER_NODE=$GPU_PER_WORKER
# Change for multinode config
MASTER_ADDR=$WORKER_0_HOST
MASTER_PORT=
NNODES=$NUM_WORKER
NODE_RANK=$WORKER_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

base_dir=$(cd `dirname $0`; pwd)
echo base_dir $base_dir


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

ds_config='{
    "train_micro_batch_size_per_gpu":16,
    "train_batch_size" : 16,
    "gradient_accumulation_steps": 2,
    "steps_per_print": 1,
    "gradient_clipping": 1.0,
    "fp16": {
      "enabled": true,
      "loss_scale": 0,
      "loss_scale_window": 1000,
      "hysteresis": 2,
      "min_loss_scale": 1
    },
    "wall_clock_breakdown": true
}'

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ${base_dir}/generate_samples_gpt.py \
       --model-parallel-size 4 \
       --num-stages 2 \
       --num-layers 30 \
       --hidden-size 3072 \
       --num-attention-heads 32 \
       --train-batch-size 1 \
       --gradient_accumulation_steps 1 \
       --batch-size 1 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --distributed-backend nccl \
       --log-interval 1 \
       --vocab-size 145608 \
       --DDP-impl torch \
       --eod-mask-loss \
       --deepspeed-pipeline \
       --deepspeed \
       --config_param "$ds_config" \
       --fp16 \
       --tokenizer-type GPT2BPETokenizer \
       --partition_method "type:ParallelTransformerLayerPiped" \
       --greedy \
       --recompute \
       $@
date
set +x
