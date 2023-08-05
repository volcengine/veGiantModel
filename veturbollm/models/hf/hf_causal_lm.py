from typing import Mapping, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast

from veturbollm.global_vars import get_args
from veturbollm.models.hf.model_wrapper import TurboHFModelWithZLoss
from veturbollm.models.hf.pretrained import state_dict_from_pretrained
from veturbollm.utils.meta_init_context import init_on_device

__all__ = ["TurboHFCausalLM"]

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class TurboHFCausalLM(TurboHFModelWithZLoss):
    """Configures a :class:`.HuggingFaceModel` around a Causal LM.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer that the model will use.
    """

    def __init__(self, tokenizer: Tokenizer):
        args = get_args()
        config = AutoConfig.from_pretrained(
            args.model.pretrained_model_name_or_path,
            trust_remote_code=args.model.trust_remote_code,
            use_auth_token=args.model.use_auth_token,
        )

        # set config overrides
        for k, v in args.model.config_overrides.items():
            if not hasattr(config, k):
                raise ValueError(f'config does not have attribute "{k}" to override ({k}: {v}).')

            attr = getattr(config, k)
            if isinstance(attr, Mapping):
                extra_keys = [_k for _k in v.keys() if _k not in attr.keys()]
                if extra_keys:
                    raise ValueError(
                        f"Config dict override got unknown keys. "
                        f"Extra keys: {extra_keys}. "
                        f"Expected (a subset of) keys: {list(attr.keys())}."
                    )
                getattr(config, k).update(v)
            else:
                setattr(config, k, v)

        if args.distributed.strategy == "fsdp" and args.rank != 0:
            init_device = torch.device("meta")
        else:
            init_device = torch.device("cpu")

        # create empty model
        with init_on_device(device=init_device):
            if args.model.enable_flash_attn:
                if "gpt" in args.model.pretrained_model_name_or_path:
                    from veturbollm.models.hf.gpt import GPTLMHeadModel

                    config.use_flash_attn = True
                    model = GPTLMHeadModel(config)
                elif "llama" in args.model.pretrained_model_name_or_path:
                    from veturbollm.models.hf.llama import LLaMAForCausalLM

                    config.use_flash_attn = True
                    model = LLaMAForCausalLM(config)
                else:
                    raise ValueError("Flash attention is only supported for GPT and LLaMA.")
            else:
                model = AutoModelForCausalLM.from_config(config)

        # load from pretrained
        if args.model.pretrained and args.rank == 0:
            state_dict = state_dict_from_pretrained(args.model.pretrained_model_name_or_path)

            # hack for load llama
            if "embed_tokens.weight" in state_dict.keys() or "tok_embeddings.weight" in state_dict.keys():
                from veturbollm.models.hf.llama import remap_state_dict_hf_llama

                state_dict = remap_state_dict_hf_llama(state_dict, model.config)
            model.load_state_dict(state_dict, strict=True)

        turbo_model = super().__init__(model=model, tokenizer=tokenizer)

        return turbo_model
