# veGiantModel
VeGiantModel is a torch based high efficient training library developed by the Applied Machine Learning team at Bytedance. This repository is for ongoing research to make giant model (such as [GPT](https://arxiv.org/abs/2005.14165), [BERT](https://arxiv.org/pdf/1810.04805.pdf) and [T5](https://arxiv.org/abs/1910.10683)) training easy, efficient, and effective. VeGiantModel builds on top of [Megatron](https://github.com/NVIDIA/Megatron-LM) and [DeepSpeed](https://github.com/microsoft/DeepSpeed), improves communication efficiency by integrating high efficient communication library [BytePs](https://github.com/bytedance/byteps) and providing customized pipline partitioning.
## initialization

```python
import veGiantModel
pipeline_parallel_size = 1
model_parallel_size = 2
veGiantModel.initialize.init_distribute(pipeline_parallel_size, model_parallel_size, init_method="env://")
mp_size = veGiantModel.distributed.get_model_parallel_world_size()
dp_size = veGiantModel.distributed.get_data_parallel_world_size()
```

## modules


```python
from veGiantModel.module import ColumnParallelLinear, RowParallelLinear

class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, config: Config):
        super().__init__()

        if self.config.use_mp_linear_in_ffn:
            assert ColumnParallelLinear is not None
            assert RowParallelLinear is not None
            self.fc1 = ColumnParallelLinear(config.dim, config.dim_ff, use_ft=False)
            self.fc2 = RowParallelLinear(config.dim_ff, config.dim, use_ft=False)
        else:
            self.fc1 = nn.Linear(config.dim, config.dim_ff)
            self.fc2 = nn.Linear(config.dim_ff, config.dim)
        self.act = Activation(config.act)
        self.dropout = nn.Dropout(config.p_drop_hidden)

    def forward(self, x) -> torch.Tensor:
        # (bsz, seq_len, dim) -> (bsz, seq_len, dim_ff / model_parallel_size) -> (bsz, seq_len, dim)
        fc1_out = self.act(self.fc1(x))
        if self.config.dropout_in_ffn:
            fc1_out = self.dropout(fc1_out)
        fc2_out = self.fc2(fc1_out)
        if self.config.use_ffn_output_dropout:
            fc2_out = self.dropout(fc2_out)
        return fc2_out
```


## Examples
### GPT Pretraining
The `examples/gpt/pretrain_gpt2_distributed.sh` scrips runs 345M parameter GPT pretraining on single 8 GPUs node. It follows largely the same as Megatron GPT script with a few notable differences. It shows good compatiblility with current megatron/Deepseed training job with little changes to adpot VeGiantModel.
