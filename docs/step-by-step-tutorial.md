# A Step-by-Step Tutorial
The goal of this tutorial is to help you run the example quickly.

## Pre-requisite
pytorch:
```
pip3 install pytorch
```

Apex:
```
git clone https://github.com/NVIDIA/apex.git
cd apex
python3 setup.py -v --cpp_ext --cuda_ext bdist_wheel
sudo pip3 install dist/*
```

BytePs:
```
pip3 install byteps
```
## Prepare data
    [GPT data preprocess](https://github.com/NVIDIA/Megatron-LM#data-preprocessing)

## Setup veGiantModel
```
git clone https://github.com/volcengine/veGiantModel.git
cd veGiantModel
git submodule update --init --recursive
```

## Modify script
Modify examples/gpt/pretrain_gpt2_distributed.sh before run
```
DATA_PATH           -- the preprocessed gpt data local folder path
CHECKPOINT_PATH     -- local path to save/load check point
MASTER_PORT         -- port number used by torch ddp
WORKER_0_PORT       -- port number for veGiantModel use for communication
WORKER_0_HOST       -- ip of the master node (single node training can use 'localhost')
NUM_WORKER          -- number of workers in the training
WORKER_RANK         -- rank of current node
GPU_PER_WORKER      -- number of GPUs per node
```

## run script
```
bash examples/gpt/pretrain_gpt2_distributed.sh
```

