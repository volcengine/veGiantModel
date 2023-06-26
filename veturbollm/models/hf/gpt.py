# Copyright (c) 2023, volcengine.
# Copyright (c) 2023, Tri Dao.

import logging
import math
import re
from collections import OrderedDict, namedtuple
from collections.abc import Sequence
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn.models.gpt_neox import remap_state_dict_hf_gpt_neox
from flash_attn.models.gptj import remap_state_dict_hf_gptj
from flash_attn.models.opt import remap_state_dict_hf_opt
from flash_attn.modules.embedding import GPT2Embeddings

from flash_attn.modules.mlp import FusedMLP, GatedMlp, Mlp
from flash_attn.ops.activations import sqrelu_fwd
from flash_attn.utils.pretrained import state_dict_from_pretrained
from torch.nn import CrossEntropyLoss

# from flash_attn.utils.generation import GenerationMixin
from transformers import GenerationConfig, GenerationMixin, GPT2Config

from veturbollm.models.hf.llama import remap_state_dict_hf_llama
from veturbollm.modules.block import Block
from veturbollm.modules.layer_norm import dropout_add_layer_norm
from veturbollm.modules.rms_norm import RMSNorm, dropout_add_rms_norm
from veturbollm.modules.mha import MHA

try:
    from flash_attn.ops.triton.mlp import FusedDenseSqreluDense
except ImportError:
    FusedDenseSqreluDense = None


logger = logging.getLogger(__name__)


def create_mixer_cls(config, layer_idx=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    softmax_scale = 1.0 if not config.scale_attn_weights else head_dim ** (-0.5)
    if config.scale_attn_by_inverse_layer_idx:
        assert layer_idx is not None
        softmax_scale /= float(layer_idx + 1)
    dwconv = getattr(config, "attn_dwconv", False)
    qkv_proj_bias = getattr(config, "qkv_proj_bias", True)
    out_proj_bias = getattr(config, "out_proj_bias", True)
    rotary_emb_dim = int(getattr(config, "rotary_emb_fraction", 0.0) * head_dim)
    rotary_emb_base = getattr(config, "rotary_emb_base", 10000.0)
    rotary_emb_scale_base = getattr(config, "rotary_emb_scale_base", None)
    rotary_emb_interleaved = getattr(config, "rotary_emb_interleaved", False)
    use_flash_attn = getattr(config, "use_flash_attn", False)
    fused_bias_fc = getattr(config, "fused_bias_fc", False)
    mha_cls = MHA
    serial_kwargs = {"fused_bias_fc": fused_bias_fc, "dwconv": dwconv}
    mixer_cls = partial(
        mha_cls,
        num_heads=config.num_attention_heads,
        qkv_proj_bias=qkv_proj_bias,
        out_proj_bias=out_proj_bias,
        dropout=config.attn_pdrop,
        softmax_scale=softmax_scale,
        causal=True,
        layer_idx=layer_idx,
        rotary_emb_dim=rotary_emb_dim,
        rotary_emb_base=rotary_emb_base,
        rotary_emb_scale_base=rotary_emb_scale_base,
        rotary_emb_interleaved=rotary_emb_interleaved,
        use_flash_attn=use_flash_attn,
        **serial_kwargs,
        **factory_kwargs,
    )
    return mixer_cls


def create_mlp_cls(config, layer_idx=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    mlp_fc1_bias = getattr(config, "mlp_fc1_bias", True)
    mlp_fc2_bias = getattr(config, "mlp_fc2_bias", True)
    fused_mlp = getattr(config, "fused_mlp", False)
    if fused_mlp:
        assert config.activation_function in ["gelu_new", "gelu_fast", "gelu_approx", "relu", "sqrelu"]
    fused_dense_sqrelu_dense = getattr(config, "fused_dense_sqrelu_dense", False)
    if fused_dense_sqrelu_dense:
        assert config.activation_function == "sqrelu", (
            "fused_dense_sqrelu_dense only " "supports approximate activation_function sqrelu"
        )
    assert not (fused_dense_sqrelu_dense and fused_mlp)
    if not fused_mlp and not fused_dense_sqrelu_dense:
        assert config.activation_function in [
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "relu",
            "sqrelu",
            "glu",
            "swiglu",
            "geglu",
        ]
        if config.activation_function in ["glu", "swiglu", "geglu"]:
            activation = (
                F.sigmoid
                if config.activation_function == "glu"
                else (F.silu if config.activation_function == "swiglu" else F.gelu)
            )
            mlp_cls = partial(
                GatedMlp,
                hidden_features=config.n_inner,
                activation=activation,
                bias1=mlp_fc1_bias,
                bias2=mlp_fc2_bias,
                **factory_kwargs,
            )
        else:
            if config.activation_function == "relu":
                activation = partial(F.relu, inplace=True)
            elif config.activation_function == "sqrelu":
                activation = sqrelu_fwd
            else:
                approximate = (
                    "tanh" if config.activation_function in ["gelu_new", "gelu_fast", "gelu_approx"] else "none"
                )
                activation = partial(F.gelu, approximate=approximate)
            mlp_cls = partial(
                Mlp,
                hidden_features=config.n_inner,
                activation=activation,
                bias1=mlp_fc1_bias,
                bias2=mlp_fc2_bias,
                **factory_kwargs,
            )
    else:
        mlp_checkpoint_lvl = getattr(config, "mlp_checkpoint_lvl", 0)
        # mlp_checkpoint_lvl could be a list, which contains the checkpoint_lvl for each layer
        if isinstance(mlp_checkpoint_lvl, Sequence):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        if fused_mlp:
            if FusedMLP is None:
                raise ImportError("fused_dense is not installed")
            activation = (
                "gelu_approx"
                if config.activation_function in ["gelu_new", "gelu_fast", "gelu_approx"]
                else config.activation_function
            )
            mlp_cls = FusedMLP
            mlp_cls = partial(
                mlp_cls,
                hidden_features=config.n_inner,
                activation=activation,
                checkpoint_lvl=mlp_checkpoint_lvl,
                bias1=mlp_fc1_bias,
                bias2=mlp_fc2_bias,
                **factory_kwargs,
            )
        elif fused_dense_sqrelu_dense:
            assert FusedDenseSqreluDense is not None
            mlp_cls = partial(
                FusedDenseSqreluDense,
                hidden_features=config.n_inner,
                checkpoint_lvl=mlp_checkpoint_lvl,
                **factory_kwargs,
            )
        else:
            raise RuntimeError("MLP type not supported")
    return mlp_cls


def create_block(config, layer_idx=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = create_mixer_cls(config, layer_idx, **factory_kwargs)
    mlp_cls = create_mlp_cls(config, layer_idx, **factory_kwargs)
    use_rms_norm = getattr(config, "rms_norm", False)
    norm_cls = partial(nn.LayerNorm if not use_rms_norm else RMSNorm, eps=config.layer_norm_epsilon, **factory_kwargs)
    # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
    residual_in_fp32 = getattr(config, "residual_in_fp32", False)
    resid_dropout1 = config.resid_pdrop if layer_idx is None or layer_idx > 0 else config.embd_pdrop
    prenorm = getattr(config, "prenorm", True)
    block = Block(
        config.hidden_size,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=prenorm,
        resid_dropout1=resid_dropout1,
        resid_dropout2=config.resid_pdrop,
        fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class GPTPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPT2Config`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    @classmethod
    def from_pretrained(
        cls, model_name, config, *args, strict=True, device=None, dtype=None, world_size=1, rank=0, **kwargs
    ):
        """
        Instantiate a GPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *args, device=device, dtype=dtype, **kwargs)
        # Load state_dict in cpu because we already initialized the model in GPU, and we don't
        # want extra stuff taking up more GPU memory
        state_dict = state_dict_from_pretrained(model_name, device="cpu", dtype=dtype)
        if model_name.startswith("gpt2"):
            state_dict = remap_state_dict_hf_gpt2(state_dict, config)
        elif model_name.startswith("facebook/opt"):
            state_dict = remap_state_dict_hf_opt(state_dict, config)
        elif model_name.startswith("EleutherAI/gpt-j-"):
            state_dict = remap_state_dict_hf_gptj(state_dict, config)
            strict = False  # We have rotary_emb.inf_freq buffers not in the GPT-J checkpoint
        elif model_name.startswith("EleutherAI/gpt-neox-"):
            state_dict = remap_state_dict_hf_gpt_neox(state_dict, config)
        elif model_name.startswith("decapoda-research/llama-"):
            state_dict = remap_state_dict_hf_llama(state_dict, config)
        else:
            raise NotImplementedError(f"Model {model_name} not supported")
        if world_size > 1:
            state_dict = shard_state_dict_tp(state_dict, config, world_size, rank)
        load_return = model.load_state_dict(state_dict, strict=strict)
        logger.info(load_return)
        return model


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


class GPTModel(GPTPreTrainedModel):
    def __init__(self, config: GPT2Config, device=None, dtype=None):
        super().__init__(config)
        factory_kwargs = {"device": device, "dtype": dtype}
        assert config.activation_function in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "relu",
            "sqrelu",
            "glu",
            "swiglu",
            "geglu",
        ]
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)
        # These 2 options are for OPT-350m
        self.prenorm = getattr(config, "prenorm", True)
        use_rms_norm = getattr(config, "rms_norm", False)
        word_embed_proj_dim = getattr(config, "word_embed_proj_dim", None)

        self.embeddings = GPT2Embeddings(
            config.hidden_size,
            vocab_size,
            config.max_position_embeddings,
            word_embed_proj_dim=word_embed_proj_dim,
            **factory_kwargs,
        )

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.layers = nn.ModuleList(
            [create_block(config, layer_idx=i, **factory_kwargs) for i in range(config.num_hidden_layers)]
        )

        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.prenorm:
            self.drop_f = nn.Dropout(config.resid_pdrop)
            norm_cls = nn.LayerNorm if not use_rms_norm else RMSNorm
            self.ln_f = norm_cls(config.hidden_size, eps=config.layer_norm_epsilon, **factory_kwargs)

        self.apply(
            partial(_init_weights, n_layer=config.num_hidden_layers, initializer_range=config.initializer_range)
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, position_ids=None, inference_params=None):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None
        mixer_kwargs = {}
        if inference_params is not None:
            mixer_kwargs["inference_params"] = inference_params
        for layer in self.layers:
            if self.prenorm:
                hidden_states, residual = layer(hidden_states, residual, mixer_kwargs=mixer_kwargs)
            else:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
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


class GPTLMHeadModel(GPTPreTrainedModel, GenerationMixin):
    def __init__(self, config: GPT2Config, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(config)
        self.main_input_name = "input_ids"
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

        self.transformer = GPTModel(config, **factory_kwargs)
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        lm_head_bias = getattr(config, "lm_head_bias", False)
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        # This option is for OPT-350m
        word_embed_proj_dim = getattr(config, "word_embed_proj_dim", None)
        embed_dim = config.n_embd if word_embed_proj_dim is None else word_embed_proj_dim
        if word_embed_proj_dim is not None:
            self.project_out = nn.Linear(config.n_embd, embed_dim, bias=False, **factory_kwargs)
        else:
            self.project_out = None
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=lm_head_bias, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(_init_weights, n_layer=config.num_hidden_layers, initializer_range=config.initializer_range)
        )
        self.tie_weights()

    def tie_weights(self):
        if self.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embeddings.word_embeddings.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.transformer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def reset_parameters(self):
        # TODO: check if this is necessary
        self.apply(
            partial(
                _init_weights, n_layer=self.config.num_hidden_layers, initializer_range=self.config.initializer_range
            )
        )

    def forward(
        self, input_ids, labels=None, position_ids=None, inference_params=None, last_token_only=False, **kwargs
    ):
        """
        inference_params: for generation. Adapted from Megatron-LM (and Apex)
        https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        last_token_only: whether to return the logit for the last token only,
            of shape (batch_size, vocab_size)
        """
        hidden_states = self.transformer(input_ids, position_ids=position_ids, inference_params=inference_params)
        if last_token_only:
            hidden_states = hidden_states[:, -1]
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
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


def shard_state_dict_tp(state_dict, config, world_size, rank):
    """Convert the state_dict of a standard GPT model to the state_dict of a GPT model."""
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    assert vocab_size % world_size == 0
    assert config.hidden_size % world_size == 0
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    assert inner_dim % world_size == 0

    def shard_first_dim(state_dict, key):
        x = state_dict[key]
        dim = x.shape[0] // world_size
        state_dict[key] = x[rank * dim : (rank + 1) * dim]

    def shard_last_dim(state_dict, key):
        x = state_dict[key]
        dim = x.shape[-1] // world_size
        state_dict[key] = x[..., rank * dim : (rank + 1) * dim]

    def shard_qkv_headdim(state_dict, key):
        x = rearrange(state_dict[key], "(three d) ... -> three d ...", three=3)
        dim = x.shape[1] // world_size
        state_dict[key] = rearrange(x[:, rank * dim : (rank + 1) * dim], "three d ... -> (three d) ...")

    shard_first_dim(state_dict, "transformer.embeddings.word_embeddings.weight")
    if "lm_head.weight" in state_dict:
        shard_first_dim(state_dict, "lm_head.weight")
    if "transformer.embeddings.position_embeddings.weight" in state_dict:
        shard_last_dim(state_dict, "transformer.embeddings.position_embeddings.weight")
    for i in range(config.num_hidden_layers):
        shard_qkv_headdim(state_dict, f"transformer.layers.{i}.mixer.Wqkv.weight")
        shard_qkv_headdim(state_dict, f"transformer.layers.{i}.mixer.Wqkv.bias")
        shard_last_dim(state_dict, f"transformer.layers.{i}.mixer.out_proj.weight")
        if rank != 0:
            state_dict.pop(f"transformer.layers.{i}.mixer.out_proj.bias")
        shard_first_dim(state_dict, f"transformer.layers.{i}.mlp.fc1.weight")
        shard_first_dim(state_dict, f"transformer.layers.{i}.mlp.fc1.bias")
        shard_last_dim(state_dict, f"transformer.layers.{i}.mlp.fc2.weight")
        if rank != 0:
            state_dict.pop(f"transformer.layers.{i}.mlp.fc2.bias")
    return state_dict


def combine_state_dicts_tp(state_dicts, config):
    """Convert the state_dict of a standard GPT model to the state_dict of a GPT model
    with tensor parallel.
    """
    world_size = len(state_dicts)
    keys = state_dicts[0].keys()
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    assert vocab_size % world_size == 0
    assert config.hidden_size % world_size == 0
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    assert inner_dim % world_size == 0

    # Sometimes the word embeddings are sharded on the 0th dim, sometimes on the 1st dim.
    # vocab_size // world_size coordinates are nonzero.
    def combine_word_embeddings(state_dicts, state_dict, key):
        dim = 0 if state_dicts[0][key].shape[0] == vocab_size // world_size else 1
        state_dict[key] = torch.cat([s[key] for s in state_dicts], dim=dim)

    def combine_dim(state_dicts, state_dict, key, dim=-1):
        if key in state_dict:
            state_dict[key] = torch.cat([s[key] for s in state_dicts], dim=dim)

    def combine_qkv_headdim(state_dicts, state_dict, key):
        if key in state_dict:
            xs = [rearrange(s[key], "(three d) ... -> three d ...", three=3) for s in state_dicts]
            state_dict[key] = rearrange(torch.cat(xs, dim=1), "three d ... -> (three d) ...")

    def combine_gated_mlp(state_dicts, state_dict, key):
        if key in state_dict:
            xs = [rearrange(s[key], "(two d) ... -> two d ...", two=2) for s in state_dicts]
            state_dict[key] = rearrange(torch.cat(xs, dim=1), "two d ... -> (two d) ...")

    state_dict = state_dicts[0].copy()  # don't modify state_dict[0] inplace
    combine_word_embeddings(state_dicts, state_dict, "transformer.embeddings.word_embeddings.weight")
    if "lm_head.weight" in state_dict:
        combine_word_embeddings(state_dicts, state_dict, "lm_head.weight")
    if "transformer.embeddings.position_embeddings.weight" in state_dict:
        combine_dim(state_dicts, state_dict, "transformer.embeddings.position_embeddings.weight", -1)
    mlp_combine_fn = (
        combine_gated_mlp if config.activation_function in ["glu", "swiglu", "geglu"] else partial(combine_dim, dim=0)
    )
    for i in range(config.num_hidden_layers):
        combine_qkv_headdim(state_dicts, state_dict, f"transformer.layers.{i}.mixer.Wqkv.weight")
        combine_qkv_headdim(state_dicts, state_dict, f"transformer.layers.{i}.mixer.Wqkv.bias")
        combine_dim(state_dicts, state_dict, f"transformer.layers.{i}.mixer.out_proj.weight", -1)
        mlp_combine_fn(state_dicts, state_dict, f"transformer.layers.{i}.mlp.fc1.weight")
        combine_dim(state_dicts, state_dict, f"transformer.layers.{i}.mlp.fc1.bias", 0)
        combine_dim(state_dicts, state_dict, f"transformer.layers.{i}.mlp.fc2.weight", -1)
    return state_dict


def remap_state_dict_hf_gpt2(state_dict, config):
    # Word embedding and position embedding
    def key_mapping_pos_emb(key):
        return re.sub(r"^wpe.", "transformer.embeddings.position_embeddings.", key)

    state_dict = OrderedDict((key_mapping_pos_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("wte.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    state_dict["lm_head.weight"] = state_dict["transformer.embeddings.word_embeddings.weight"]

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^ln_f.(weight|bias)", r"transformer.ln_f.\1", key)
        key = re.sub(r"^h.(\d+).ln_(1|2).(weight|bias)", r"transformer.layers.\1.norm\2.\3", key)
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    for d in range(config.num_hidden_layers):
        W1 = state_dict.pop(f"h.{d}.mlp.c_fc.weight")
        state_dict[f"transformer.layers.{d}.mlp.fc1.weight"] = W1.t()
        W2 = state_dict.pop(f"h.{d}.mlp.c_proj.weight")
        state_dict[f"transformer.layers.{d}.mlp.fc2.weight"] = W2.t()

    def key_mapping_mlp(key):
        key = re.sub(r"^h.(\d+).mlp.c_fc.bias", r"transformer.layers.\1.mlp.fc1.bias", key)
        key = re.sub(r"^h.(\d+).mlp.c_proj.bias", r"transformer.layers.\1.mlp.fc2.bias", key)
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for d in range(config.num_hidden_layers):
        state_dict.pop(f"h.{d}.attn.bias")  # We don't store this bias
        Wqkv = state_dict.pop(f"h.{d}.attn.c_attn.weight")
        state_dict[f"transformer.layers.{d}.mixer.Wqkv.weight"] = Wqkv.t()
        Wout = state_dict.pop(f"h.{d}.attn.c_proj.weight")
        state_dict[f"transformer.layers.{d}.mixer.out_proj.weight"] = Wout.t()

    def key_mapping_attn(key):
        key = re.sub(r"^h.(\d+).attn.c_attn.bias", r"transformer.layers.\1.mixer.Wqkv.bias", key)
        key = re.sub(r"^h.(\d+).attn.c_proj.bias", r"transformer.layers.\1.mixer.out_proj.bias", key)
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def remap_state_dict_megatron(state_dict, config):
    def key_mapping_transformer(key):
        key = re.sub(r"^language_model.encoder.", "transformer.", key)
        key = re.sub(r"^language_model.", "transformer.", key)
        return key

    state_dict = OrderedDict((key_mapping_transformer(k), v) for k, v in state_dict.items())

    # Word embedding and position embedding
    def key_mapping_pos_emb(key):
        return re.sub(r"^wpe.", "transformer.embeddings.position_embeddings.", key)

    state_dict = OrderedDict((key_mapping_pos_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("transformer.embedding.word_embeddings.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(word_embeddings.shape[0] / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    state_dict["lm_head.weight"] = state_dict["transformer.embeddings.word_embeddings.weight"]

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^transformer.final_layernorm.(weight|bias)", r"transformer.ln_f.\1", key)
        key = re.sub(
            r"^transformer.layers.(\d+).input_layernorm.(weight|bias)", r"transformer.layers.\1.norm1.\2", key
        )
        key = re.sub(
            r"^transformer.layers.(\d+).post_attention_layernorm.(weight|bias)", r"transformer.layers.\1.norm2.\2", key
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(
            r"^transformer.layers.(\d+).mlp.dense_h_to_4h.(weight|bias)", r"transformer.layers.\1.mlp.fc1.\2", key
        )
        key = re.sub(
            r"^transformer.layers.(\d+).mlp.dense_4h_to_h.(weight|bias)", r"transformer.layers.\1.mlp.fc2.\2", key
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    def key_mapping_attn(key):
        key = re.sub(
            r"^transformer.layers.(\d+).self_attention.rotary_emb.inv_freq",
            r"transformer.layers.\1.mixer.rotary_emb.inv_freq",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).self_attention.query_key_value.(weight|bias)",
            r"transformer.layers.\1.mixer.Wqkv.\2",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).self_attention.dense.(weight|bias)",
            r"transformer.layers.\1.mixer.out_proj.\2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())
    # Megatron stores Wqkv as ((nheads 3 headdim), hidden_dim)
    # while we store Wqkv as ((3 nheads headdim), hidden_dim)
    headdim = config.hidden_size // config.num_attention_heads
    for d in range(config.num_hidden_layers):
        Wqkv = state_dict.pop(f"transformer.layers.{d}.mixer.Wqkv.weight")
        state_dict[f"transformer.layers.{d}.mixer.Wqkv.weight"] = rearrange(
            Wqkv, "(nheads three headdim) ... -> (three nheads headdim) ...", three=3, headdim=headdim
        )
        bqkv = state_dict.pop(f"transformer.layers.{d}.mixer.Wqkv.bias")
        state_dict[f"transformer.layers.{d}.mixer.Wqkv.bias"] = rearrange(
            bqkv, "(nheads three headdim) -> (three nheads headdim)", three=3, headdim=headdim
        )

    return state_dict
