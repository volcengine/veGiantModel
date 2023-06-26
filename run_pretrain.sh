pip install -r requirements.txt

# 用于解决 huggingface 下载模型时缓存问题，建议先在共享文件系统中下载模型，然后将文件夹链接进来，或者直接在 configs 中进行配置
# ln -s /vepfs/huggingface_cache ~/.cache/huggingface

# Tracking SDK
wget https://ml-platform-public-examples-cn-beijing.tos-cn-beijing.volces.com/python_sdk_installer/volcengine_ml_platform-1.0.13-py3-none-any.whl && pip install volcengine_ml_platform-1.0.13-py3-none-any.whl -i https://mirrors.ivolces.com/pypi/simple


export PYTHONPATH=./:$PYTHONPATH
export GPUS_PER_NODE=${MLP_WORKER_GPU:-8}
export NNODES=${MLP_WORKER_NUM:-1}
export NODE_RANK=${MLP_ROLE_INDEX:-0}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-127.0.0.1}
export MASTER_PORT=${MLP_WORKER_0_PORT:-1234}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS scripts/pretrain/train.py --config_file configs/llama65b.yaml
