from typing import Mapping, Union

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast

from veturbollm.global_vars import get_args
from veturbollm.models.hf.model_wrapper import TurboHFModelWithZLoss

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

        if args.distributed.strategy == "fsdp":
            # if dist.get_rank() % torch.cuda.device_count() == 0:
            if dist.get_rank() == 0:
                init_device = "cpu"
            else:
                # TODO: 65B model need meta device, but this action will make process hang
                init_device = "meta"
        else:
            init_device = "cpu"

        if init_device == "cpu" and args.model.pretrained:
            if args.model.enable_flash_attn:
                from veturbollm.models.hf.gpt import GPTLMHeadModel

                if "llama" in args.model.pretrained_model_name_or_path:
                    from veturbollm.models.hf.llama import llama_config_to_gpt2_config

                    config = llama_config_to_gpt2_config(config)
                config.use_flash_attn = True
                model = GPTLMHeadModel.from_pretrained(args.model.pretrained_model_name_or_path, config)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model.pretrained_model_name_or_path,
                    trust_remote_code=args.model.trust_remote_code,
                    use_auth_token=args.model.use_auth_token,
                    config=config,
                )
        else:
            init_device = "meta"
            from veturbollm.utils.meta_init_context import init_empty_weights

            with init_empty_weights():
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
        turbo_model = super().__init__(model=model, tokenizer=tokenizer)

        return turbo_model
