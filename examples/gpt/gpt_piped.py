import torch

from megatron import get_args, mpu

from megatron.model.language_model import parallel_lm_logits, Embedding
from megatron.model.transformer import ParallelTransformerLayer
from megatron.model.transformer import LayerNorm
from megatron.model.gpt2_model import gpt2_attention_mask_func
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from megatron.module import MegatronModule
from megatron.utils import get_ltor_masks_and_position_ids

from deepspeed.pipe import LayerSpec, TiedLayerSpec
from megatron import get_tokenizer
from veGiantModel.engine.module import VeGiantModule
import veGiantModel

class GPTModelPiped(VeGiantModule):
    def __init__(self):
        args = get_args()
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.tokenizer = get_tokenizer()
        self.parallel_output = True

        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size

        self.init_method = init_method_normal(args.init_method_std)
        self.scale_init_method = scaled_init_method_normal(args.init_method_std,
                                                           args.num_layers)

        self.num_tokentypes = 0

        layers = []
        layers.append(lambda x: self._get_batch(x))
        layers.append(TiedLayerSpec("SharedEmbedding",
                                    EmbeddingPiped,
                                    self.hidden_size,
                                    args.padded_vocab_size,
                                    args.max_position_embeddings,
                                    args.hidden_dropout,
                                    self.init_method,
                                    self.num_tokentypes,
                                    tied_weight_attr='embedding_weight'))

        layers.append(lambda x: (x[0].transpose(0, 1).contiguous(), x[1]))

        for i in range(self.num_layers):
            layers.append(LayerSpec(ParallelTransformerLayerPiped,
                                    gpt2_attention_mask_func,
                                    self.init_method,
                                    self.scale_init_method,
                                    i+1))

        layers.append(lambda x: (x[0].transpose(0, 1).contiguous()))

        layers.append(LayerSpec(LayerNorm, args.hidden_size, eps=args.layernorm_epsilon))

        layers.append(TiedLayerSpec("SharedEmbedding",
                                    LMLogitsPiped,
                                    self.hidden_size,
                                    args.padded_vocab_size,
                                    self.init_method,
                                    tied_weight_attr='embedding_weight'))

        super().__init__(layers=layers,
                         num_stages = args.num_stages, 
                         partition_method=args.partition_method,
                         grid=veGiantModel.distributed.get_grid(),
                         loss_fn=self.loss_fn)
        

    # Data Preprocessing, copied from pretrain_gpt2.py
    def _get_batch(self, data):
        """Generate a batch"""
        args = get_args()
        # Unpack.
        tokens = data

        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        return (tokens.to(device="cuda"),
                position_ids.to(device="cuda"),
                attention_mask.to(device="cuda"))

    def loss_fn(self, inputs, data):
        tokens = data[0]
        target = data[1]
        args = get_args()
        _, loss_mask, _ = get_ltor_masks_and_position_ids(
            tokens,
            self.tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        if self.fp16_lm_cross_entropy:
            assert inputs.dtype == torch.half
            loss = mpu.vocab_parallel_cross_entropy(inputs, target)
        else:
            loss = mpu.vocab_parallel_cross_entropy(inputs.float(), target)
        loss_mask = loss_mask.view(-1)
        loss_avg = torch.sum(loss.view(-1) * loss_mask) / loss_mask.sum()
        if loss.dtype == torch.half:
            loss_avg = loss_avg.half()

        return loss_avg

    def batch_fn(self, batch, is_train:bool):
        if batch is not None:
            data = {'text': torch.tensor(batch['text'].numpy())}
        else:
            data = None

        keys = ['text']
        datatype = torch.int64

        data_b = mpu.broadcast_data(keys, data, datatype)

        tokens_ = data_b['text'].long()
        tokens_write = tokens_
        labels = tokens_[:, 1:].contiguous()
        tokens_ = tokens_[:, :-1].contiguous()
        tokens_2 = torch.unsqueeze(tokens_, 0)
        data2 = torch.cat((tokens_2, labels[None, :, :]), dim=0)
        data = []
        data.append(tokens_)
        data.append(data2)
        return data

class LMLogitsPiped(MegatronModule):
    def __init__(self, hidden_size, vocab_size, init_method):
        super().__init__()
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)
        self.embedding_weight = self.word_embeddings.weight

    def forward(self, lm_output):
        return parallel_lm_logits(lm_output, self.embedding_weight, True)


class EmbeddingPiped(Embedding):
    def __init__(self,
                hidden_size,
                vocab_size,
                max_sequence_length,
                embedding_dropout_prob,
                init_method,
                num_tokentypes=0):
        super().__init__(hidden_size,
                        vocab_size,
                        max_sequence_length,
                        embedding_dropout_prob,
                        init_method,
                        num_tokentypes)
        self.embedding_weight = self.word_embeddings.weight

    def forward(self, inputs):
        input_ids, position_ids, attention_mask = inputs
        return super().forward(input_ids, position_ids, None), attention_mask

class ParallelTransformerLayerPiped(ParallelTransformerLayer):
    def __init__(self,
                attention_mask_func,
                init_method,
                output_layer_init_method,
                layer_number):
        super().__init__(attention_mask_func,
                         init_method,
                         output_layer_init_method,
                         layer_number)

    def forward(self, inputs):
        hidden_states, attention_mask = inputs
        return (super().forward(hidden_states, attention_mask),
                attention_mask)