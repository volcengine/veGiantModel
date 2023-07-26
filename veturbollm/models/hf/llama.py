# Copyright (c) 2023, Tri Dao.

import json
import logging
import math
import re
from collections import OrderedDict, namedtuple
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import GenerationConfig, GenerationMixin, LlamaConfig

from veturbollm.models.hf.pretrained import state_dict_from_pretrained
from veturbollm.modules.block import Block
from veturbollm.modules.embedding import GPT2Embeddings
from veturbollm.modules.layer_norm import dropout_add_layer_norm
from veturbollm.modules.mha import MHA
from veturbollm.modules.mlp import GatedMLP
from veturbollm.modules.rms_norm import RMSNorm, dropout_add_rms_norm

logger = logging.getLogger(__name__)


def remap_state_dict_hf_llama(state_dict, config: LlamaConfig):
    llama_v2 = False
    if "rope.freqs" in state_dict:
        freqs = state_dict.pop("rope.freqs")
        llama_v2 = True

    def key_mapping_layers(key):
        return f"transformer.{key}" if not key.startswith("output.") else key

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # Word embedding
    def key_mapping_emb(key):
        key = re.sub(r"^transformer.model.embed_tokens.", "transformer.embeddings.word_embeddings.", key)
        key = re.sub(r"^transformer.tok_embeddings.", "transformer.embeddings.word_embeddings.", key)
        return key

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("transformer.embeddings.word_embeddings.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(word_embeddings.shape[0] / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )

    # LayerNorm
    def key_mapping_ln(key):
        # llama v1
        key = re.sub(r"^transformer.model.norm.", r"transformer.ln_f.", key)
        key = re.sub(r"^transformer.model.layers.(\d+).input_layernorm.", r"transformer.layers.\1.norm1.", key)
        key = re.sub(
            r"^transformer.model.layers.(\d+).post_attention_layernorm.", r"transformer.layers.\1.norm2.", key
        )

        # llama v2
        key = re.sub(r"^transformer.norm.", r"transformer.ln_f.", key)
        key = re.sub(r"^transformer.layers.(\d+).attention_norm.", r"transformer.layers.\1.norm1.", key)
        key = re.sub(r"^transformer.layers.(\d+).ffn_norm.", r"transformer.layers.\1.norm2.", key)
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    for l in range(config.num_hidden_layers):
        if llama_v2:
            gate_proj = state_dict.pop(f"transformer.layers.{l}.feed_forward.w1.weight")
            up_proj = state_dict.pop(f"transformer.layers.{l}.feed_forward.w3.weight")
        else:
            gate_proj = state_dict.pop(f"transformer.model.layers.{l}.mlp.gate_proj.weight")
            up_proj = state_dict.pop(f"transformer.model.layers.{l}.mlp.up_proj.weight")
        # Our ordering is different
        state_dict[f"transformer.layers.{l}.mlp.fc1.weight"] = torch.cat([up_proj, gate_proj], dim=0)

    def key_mapping_mlp(key):
        key = re.sub(r"^transformer.model.layers.(\d+).mlp.down_proj.", r"transformer.layers.\1.mlp.fc2.", key)
        key = re.sub(r"^transformer.layers.(\d+).feed_forward.w2.", r"transformer.layers.\1.mlp.fc2.", key)
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for l in range(config.num_hidden_layers):
        state_dict[f"transformer.layers.{l}.mixer.rotary_emb.inv_freq"] = freqs
        if llama_v2:
            Wq = state_dict.pop(f"transformer.layers.{l}.attention.wq.weight")
            Wk = state_dict.pop(f"transformer.layers.{l}.attention.wk.weight")
            Wv = state_dict.pop(f"transformer.layers.{l}.attention.wv.weight")
            state_dict[f"transformer.layers.{l}.mixer.Wqkv.weight"] = torch.cat([Wq, Wk, Wv], dim=0)
        else:
            Wq = state_dict.pop(f"transformer.model.layers.{l}.self_attn.q_proj.weight")
            Wk = state_dict.pop(f"transformer.model.layers.{l}.self_attn.k_proj.weight")
            Wv = state_dict.pop(f"transformer.model.layers.{l}.self_attn.v_proj.weight")
            state_dict[f"transformer.layers.{l}.mixer.Wqkv.weight"] = torch.cat([Wq, Wk, Wv], dim=0)

    def key_mapping_attn(key):
        # v2
        key = re.sub(r"^transformer.layers.(\d+).attention.wo.", r"transformer.layers.\1.mixer.out_proj.", key)
        # v1
        key = re.sub(
            r"^transformer.model.layers.(\d+).self_attn.o_proj.", r"transformer.layers.\1.mixer.out_proj.", key
        )
        key = re.sub(
            r"^transformer.model.layers.(\d+).self_attn.rotary_emb.inv_freq",
            r"transformer.layers.\1.mixer.rotary_emb.inv_freq",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())
    try:
        state_dict["lm_head.weight"] = state_dict.pop("transformer.lm_head.weight")
    except:
        state_dict["lm_head.weight"] = state_dict.pop("output.weight")
    return state_dict


def state_dicts_from_checkpoint(checkpoint_path: str, model_name: str) -> dict:
    # Need to sort, otherwise we mess up the ordering and the weights are wrong
    return [
        torch.load(path, map_location="cpu")
        for path in sorted((Path(checkpoint_path) / model_name).glob("consolidated.*.pth"))
    ]


def create_mixer_cls(config: LlamaConfig, layer_idx=None, device=None, dtype=None):
    head_dim = config.hidden_size // config.num_attention_heads
    softmax_scale = head_dim ** (-0.5)

    use_flash_attn = getattr(config, "use_flash_attn", False)
    fused_bias_fc = getattr(config, "fused_bias_fc", False)
    mixer_cls = partial(
        MHA,
        num_heads=config.num_attention_heads,
        qkv_proj_bias=False,
        out_proj_bias=False,
        softmax_scale=softmax_scale,
        causal=True,
        layer_idx=layer_idx,
        rotary_emb_dim=head_dim,
        rotary_emb_interleaved=True,
        use_flash_attn=use_flash_attn,
        fused_bias_fc=fused_bias_fc,
        device=device,
        dtype=dtype,
    )
    return mixer_cls


def create_mlp_cls(config: LlamaConfig, layer_idx=None, device=None, dtype=None):
    mlp_cls = partial(
        GatedMLP,
        hidden_features=config.intermediate_size,
        activation=F.silu,
        bias1=False,
        bias2=False,
        device=device,
        dtype=dtype,
    )
    return mlp_cls


def create_block(config: LlamaConfig, layer_idx=None, device=None, dtype=None):
    mixer_cls = create_mixer_cls(config, layer_idx, device=device, dtype=dtype)
    mlp_cls = create_mlp_cls(config, layer_idx, device=device, dtype=dtype)
    norm_cls = partial(RMSNorm, eps=config.rms_norm_eps, device=device, dtype=dtype)

    residual_in_fp32 = getattr(config, "residual_in_fp32", False)
    prenorm = getattr(config, "prenorm", True)
    block = Block(
        config.hidden_size,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=prenorm,
        resid_dropout1=0.0,
        resid_dropout2=0.0,
        fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))


class LLaMAPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config: LlamaConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, LlamaConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `LlamaConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name, config, *args, strict=True, device=None, dtype=None, **kwargs):
        """
        Instantiate a GPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *args, device=device, dtype=dtype, **kwargs)
        # Load state_dict in cpu because we already initialized the model in GPU, and we don't
        # want extra stuff taking up more GPU memory
        state_dict = state_dict_from_pretrained(model_name, device="cpu", dtype=dtype)
        state_dict = remap_state_dict_hf_llama(state_dict, config)

        load_return = model.load_state_dict(state_dict, strict=strict)
        logger.info(load_return)
        return model


class LLaMAModel(LLaMAPreTrainedModel):
    def __init__(self, config: LlamaConfig, device=None, dtype=None):
        super().__init__(config)

        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)
        self.prenorm = getattr(config, "prenorm", True)
        self.embeddings = GPT2Embeddings(
            config.hidden_size,
            vocab_size,
            max_position_embeddings=0,
            device=device,
            dtype=dtype,
        )

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.layers = nn.ModuleList(
            [create_block(config, layer_idx=i, device=device, dtype=dtype) for i in range(config.num_hidden_layers)]
        )

        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.prenorm:
            # warning: meta not release yet
            self.drop_f = nn.Dropout(0.0)
            self.ln_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None
        for layer in self.layers:
            if self.prenorm:
                hidden_states, residual = layer(hidden_states, residual)
            else:
                hidden_states = layer(hidden_states)
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_f(hidden_states)
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = dropout_add_rms_norm if isinstance(self.ln_f, RMSNorm) else dropout_add_layer_norm
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    residual,
                    self.ln_f.weight,
                    self.ln_f.bias,
                    self.drop_f.p if self.training else 0.0,
                    self.ln_f.eps,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
        return hidden_states


class LLaMAForCausalLM(LLaMAPreTrainedModel, GenerationMixin):
    def __init__(self, config: LlamaConfig, device=None, dtype=None):
        super().__init__(config)
        self.main_input_name = "input_ids"
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

        self.transformer = LLaMAModel(config, device=device, dtype=dtype)
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        lm_head_bias = getattr(config, "lm_head_bias", False)
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple

        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=lm_head_bias, device=device, dtype=dtype)

        # Initialize weights and apply final processing
        # self.apply(
        #     partial(_init_weights, n_layer=config.num_hidden_layers, initializer_range=config.initializer_range)
        # )
        self.tie_weights()

        # set decoder cls
        self.decoder_cls = Block

    def tie_weights(self):
        if self.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embeddings.word_embeddings.weight

    def reset_parameters(self):
        # TODO: check if this is necessary
        self.apply(
            partial(
                _init_weights, n_layer=self.config.num_hidden_layers, initializer_range=self.config.initializer_range
            )
        )

    def forward(
        self, input_ids, labels=None, position_ids=None, past_key_values=None, last_token_only=False, **kwargs
    ):
        """
        last_token_only: whether to return the logit for the last token only,
            of shape (batch_size, vocab_size)
        """

        batch_size, seq_length = input_ids.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        hidden_states = self.transformer(input_ids, position_ids=position_ids)
        if last_token_only:
            hidden_states = hidden_states[:, -1]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "loss"])
        return CausalLMOutput(logits=lm_logits, loss=loss)

    def load_state_dict(self, state_dict, strict=True):
        # Remapping from our checkpoints that used a different ordering of layers in the block
        # Previous: Attn / MLP -> Dropout -> Add -> LN
        # Current: Dropout -> Add -> LN -> Attn / MLP
        if "transformer.ln_0.weight" in state_dict:
            n_layers = len(self.transformer.layers)
            ln_weight = state_dict.pop(f"transformer.layers.{n_layers - 1}.norm2.weight")
            ln_bias = state_dict.pop(f"transformer.layers.{n_layers - 1}.norm2.bias")
            state_dict["transformer.ln_f.weight"] = ln_weight
            state_dict["transformer.ln_f.bias"] = ln_bias
            for l in reversed(range(n_layers)):
                ln_weight = state_dict.pop(f"transformer.layers.{l}.norm1.weight")
                ln_bias = state_dict.pop(f"transformer.layers.{l}.norm1.bias")
                state_dict[f"transformer.layers.{l}.norm2.weight"] = ln_weight
                state_dict[f"transformer.layers.{l}.norm2.bias"] = ln_bias
                if l > 0:
                    ln_weight = state_dict.pop(f"transformer.layers.{l - 1}.norm2.weight")
                    ln_bias = state_dict.pop(f"transformer.layers.{l - 1}.norm2.bias")
                    state_dict[f"transformer.layers.{l}.norm1.weight"] = ln_weight
                    state_dict[f"transformer.layers.{l}.norm1.bias"] = ln_bias
            ln_weight = state_dict.pop("transformer.ln_0.weight")
            ln_bias = state_dict.pop("transformer.ln_0.bias")
            state_dict[f"transformer.layers.0.norm1.weight"] = ln_weight
            state_dict[f"transformer.layers.0.norm1.bias"] = ln_bias
        return super().load_state_dict(state_dict, strict=strict)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
            }
        )
        return model_inputs

    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation
        if "GenerationMixin" in str(self.prepare_inputs_for_generation.__func__):
            return False
        return True

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        model_embeds = self._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        self.vocab_size = new_num_tokens

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.transformer.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.transformer.embeddings.word_embeddings = new_embeddings

        # if word embeddings are not tied, make sure that lm head is resized as well
        if not self.config.tie_word_embeddings:
            old_lm_head = self.lm_head
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            self.lm_head = new_lm_head

        return self.transformer.embeddings

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        # initialize all new embeddings (in particular added tokens)
        _init_weights(new_embeddings, n_layer=0)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings

    def _get_resized_lm_head(
        self, old_lm_head: nn.Linear, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
    ) -> nn.Linear:
        if new_num_tokens is None:
            return old_lm_head

        old_num_tokens, old_lm_head_dim = (
            old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
        )

        if old_num_tokens == new_num_tokens:
            return old_lm_head

        if not isinstance(old_lm_head, nn.Linear):
            raise TypeError(
                f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}. You"
                " should either use a different resize function or make sure that `old_lm_head` are an instance of"
                f" {nn.Linear}."
            )

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None
        new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias)
        new_lm_head = new_lm_head.to(old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)

        # initialize new lm head (in particular added tokens)
        _init_weights(new_lm_head, n_layer=0)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

        return new_lm_head
