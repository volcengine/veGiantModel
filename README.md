

# veturbo-llm

此存储库包含用于训练、微调、评估和部署 LLM 的代码，以便快速使用火山引擎机器学习平台进行高效训练和推理。该代码库旨在易于高效、灵活地对最新技术进行快速试验，提供广大用户一个行之有效的最佳实践。

*本项目尚处于研发阶段*

## 运行环境

我们主要基于以下版本进行基础研发：

- Python >= 3.6
- PyTorch >= 2.0
- transformers >= 4.0.0

请使用火山引擎机器学习平台预置的镜像进行训练：

- vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:2.0.0

配置任务相关的软件环境可以通过 `pip` 命令进行安装：

```bash
pip install -r requirements.txt
```


## 快速开始
我们将在此提供了快速开始的例子，以便您快速上手。

### 预训练

我们提供了训练代码以支持不同规模的模型训练，如需要查看代码请参考 [scripts/pretrain](scripts/pretrain) 目录下的训练代码。

为了简化快速上手的过程，我们提供了基于火山引擎机器学习平台命令行工具的配置文件，您只需要修改配置文件中的资源组即可启动训练任务。

我们提供了以下配置文件，阅读配置文件可以帮助您快速上手：
- job_llama_7b.yaml
- job_llama_65b.yaml

以下是一个启动预训练任务的例子：
```bash
volc ml_task submit --conf job_llama_7b.yaml
```

## 训练性能

我们在 llama 模型上进行了测试，以下是一些测试结果：


| Model      | GPU Type  | Card Nums | miniBS | GA | TFLOPs | Throughput(tokens/gpu/s)    |
| ---------- | --------- | --------- | ------ | -- | ------ | --------------------------- |
| llama-7b   | A100-80G  | 32        | 4      | 4  | 193.9  | 4609                        |
| llama-7b   | A100-80G  | 512       | 16     | 1  | 205.5  | 4888                        |
| llama-13b  | A100-80G  | 32        | 4      | 4  | 200.5  | 2486                        |
| llama-13b  | A100-80G  | 64        | 4      | 4  | 201.6  | 2501                        |
| llama-65b  | A800-80G  | 64        | 12     | 1  | 176.0  | 440                         |
| llama-65b  | A800-80G  | 128       | 12     | 1  | 177.4  | 443                         |
| llama-65b  | A800-80G  | 256       | 12     | 1  | 173.7  | 434                         |
| llama-65b  | A800-80G  | 512       | 12     | 1  | 171.9  | 430                         |
| llama-65b  | A800-80G  | 512       | 16     | 1  | 181.4  | 454                         |

## 特性
### 火山引擎机器学习功能集成
我们集成了实验管理的功能用于跟踪实验进度。给 veturbo-llm 任务启动实验管理功能只需要在启动任务的时候增加以下环境变量：

```bash
export MLP_TRACKING_ENABLE=true
export MLP_TRACKING_PROJECT_NAME=veturbo-llm
export VOLC_ACCESSKEY=xxx
export VOLC_SECRETKEY=xxx
export VOLC_REGION=cn-beijing
```

这些环境变量也可以通过配置文件进行配置，例如在 config.yaml 中增加：

```yaml
Envs:
  - Name: "MLP_TRACKING_ENABLE"
    Value: "true"
  - Name: "MLP_TRACKING_PROJECT_NAME"
    Value: "veturbo-llm"
  - ...
```

![实验管理](misc/imgs/exp-manage.jpeg)


### 基于 FSDP 的分布式训练策略
我们在并行策略上主要支持了 FSDP 方式来支持多机多卡训练。当前主流的训练框架例如 Megatron-LM、Colossal AI 实现了例如张量并行、流水线并行等并行策略，这些策略的实现对模型并行性能有了更好的把控，但是却牺牲了算法工程师的灵活性。我们认为 FSDP 方式是一个更好的选择，它可以在不牺牲灵活性的前提下，实现高效的多机多卡训练。相比于让算法工程师去学习并行策略下的模型实现，FSDP 可以让算法工程师几乎保持与传统 DDP 相同的开发体验。

### 高性能算子/算法支持
在模型层面，veturbo-llm 主要支持了以下几个方面的优化：
- Flash Attention 算子；
- Fuse-LayerNorm 算子；

## 正在进行中的
我们正在完成更多的功能支持，包括但不限于：
- [ ] 模型保存与评估;
- [ ] 更多优化算法支持：alibi 等;
- [ ] 增加对 Megatron-LM 训练支持；
- [ ] 提供部署模块，并支持一键部署到火山引擎机器学习平台；

## 相关项目

我们从以下项目获得了非常多的灵感，在此表示感谢：
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [huggingface/transformers](https://github.com/huggingface/transformers)
- [mosaicml/llm-foundry](https://github.com/mosaicml/llm-foundry)


