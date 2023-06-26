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
        cfg (DictConfig): An omegaconf dictionary used to configure the model:
            cfg.pretrained_model_name_or_path (str): The name of or local path to
                the HF Causal LM (e.g., `gpt2` to instantiate a GPT2LMHeadModel).
            cfg.config_overrides (dict, optional): An optional dictionary of keyword
                arguments that override the default configuration associated with
                cfg.pretrained_model_name_or_path.
            cfg.pretrained (bool): Whether to instantiate the model with pre-trained
                weights coming from cfg.pretrained_model_name_or_path. If ``True``,
                cfg.config_overrides must be compatible with the pre-trained weights.
            cfg.init_device ('cpu' | 'meta'): Which device, 'cpu' or 'meta', to
                initialize the model on. Currently, `meta` is only supported when
                cfg.pretrained is ``False``. Default: ``'cpu'``.
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
                init_device = "cpu"
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
            if args.model.enable_flash_attn:
                if "gpt2" in args.model.pretrained_model_name_or_path:
                    # from flash_attn.models.gpt import GPTLMHeadModel
                    from veturbollm.models.hf.gpt import GPTLMHeadModel

                    config.use_flash_attn = True
                    model = GPTLMHeadModel(config, device=init_device)
                elif "llama" in args.model.pretrained_model_name_or_path:
                    from veturbollm.models.hf.gpt import GPTLMHeadModel
                    from veturbollm.models.hf.llama import llama_config_to_gpt2_config

                    config = llama_config_to_gpt2_config(config)
                    config.use_flash_attn = True
                    model = GPTLMHeadModel(config, device=init_device)
                else:
                    raise ValueError("Flash attention is only supported for GPT and LLAMA.")
            else:
                model = AutoModelForCausalLM.from_config(config, device=init_device)
        turbo_model = super().__init__(model=model, tokenizer=tokenizer)

        return turbo_model
