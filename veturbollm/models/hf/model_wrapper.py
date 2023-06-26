"""Re-usable :class:`.ComposerModel` for LLM HF Models."""

from __future__ import annotations

import inspect
import logging
from collections import UserDict
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import transformers
from torchmetrics import Metric
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

log = logging.getLogger(__name__)


class TurboHFModelWithZLoss(nn.Module):
    """Wrapper around TurboHFModel.

    This adds z-loss, which is used in some training contexts,
    and is a convenient way to patch features that are generically
    useful for HF models.
    See use of z_loss in PaLM: https://arxiv.org/abs/2204.02311v3, Section 5.
    Also, from https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666:
        Two uses of z_loss are:
        - To keep the logits from drifting too far from zero, which can cause
            unacceptable roundoff errors in bfloat16.
        - To encourage the logits to be normalized log-probabilities.

    Handles preparation for FSDP wrapping.
    """

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: Optional[Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]] = None,
        use_logits: Optional[bool] = False,
        metrics: Optional[List[Metric]] = None,
        eval_metrics: Optional[List[Metric]] = None,
        shift_labels: Optional[bool] = None,
        allow_embedding_resizing: bool = False,
        z_loss: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        self.model_forward_args = inspect.getfullargspec(self.model.forward).args
        self.tokenizer = tokenizer

        if self.tokenizer is None:
            log.warning(
                "The tokenizer was not provided. This means the tokenizer config will not be saved in the checkpoint."
            )

        if tokenizer is not None and self.config.vocab_size < len(tokenizer):
            if allow_embedding_resizing:
                # when the embedding size is smaller than the tokenizer vocab size,
                # the embeddings should get resized to match the tokenizer vocab size
                log.warning(
                    f"The number of tokens in the tokenizer is greater than the number of tokens in the model."
                    f" This would cause an error during training."
                    f" Resizing the model embeddings to {len(tokenizer)} from {self.config.vocab_size}."
                )
                self.model.resize_token_embeddings(len(tokenizer))
            else:
                raise ValueError(
                    f"The number of tokens in the tokenizer is greater than the number of tokens in the model."
                    f" This would cause an error during training."
                    f" You can resize the model embeddings to {len(tokenizer)} from {self.config.vocab_size}"
                    f" by calling `model.resize_token_embeddings(len(tokenizer))` before calling the `HuggingFaceModel`"
                    f" constructor, or pass `allow_embedding_resizing=True` to have it done automatically."
                )
        elif tokenizer is not None and self.config.vocab_size > len(tokenizer):
            # when the embedding size is greater than the tokenizer vocab size,
            # the embeddings do not _need_ to be resized to match the tokenizer vocab size,
            # and should be done by the user if desired
            log.info(
                f"The number of tokens in the tokenizer is less than the number of tokens in the model."
                f" You may want to resize the model embeddings to {len(tokenizer)} from {self.config.vocab_size}"
                f" by calling `model.resize_token_embeddings(len(tokenizer))` before calling the `HuggingFaceModel`"
                f" constructor. The vocab size is sometimes intentionally set to a multiple of 32 or 64 to improve"
                f" performance."
            )

        self.use_logits = use_logits

        self.train_metrics: Optional[Dict] = None
        self.val_metrics: Optional[Dict] = None

        if eval_metrics is not None:
            self.val_metrics = {metric.__class__.__name__: metric for metric in eval_metrics}
        if metrics is not None:
            self.train_metrics = {metric.__class__.__name__: metric for metric in metrics}
            # if eval_metrics is None, use the same metrics as train_metrics
            if eval_metrics is None:
                self.val_metrics = {metric.__class__.__name__: metric for metric in metrics}

        self.labels: Optional[torch.Tensor] = None  # set in eval_forward() if exists

        self.z_loss = float(z_loss)
        if self.z_loss < 0.0:
            raise ValueError(f"z_loss(={z_loss}) cannot be negative.")

        self.model_forward_args = inspect.getfullargspec(self.model.forward).args

        # Record model config
        model_config = self.model.config
        self.hidden_size = getattr(model_config, model_config.attribute_map.get("hidden_size", "hidden_size"))
        self.num_hidden_layers = getattr(
            model_config, model_config.attribute_map.get("num_hidden_layers", "num_hidden_layers")
        )
        self.max_position_embeddings = getattr(
            model_config,
            model_config.attribute_map.get("max_position_embeddings", "max_position_embeddings"),
        )
        self.num_attention_heads = getattr(
            model_config, model_config.attribute_map.get("num_attention_heads", "num_attention_heads")
        )
        self.vocab_size = len(tokenizer)
        self.n_active_params = sum(p.numel() for p in self.model.parameters())

    def forward(self, batch):
        if isinstance(batch, dict) or isinstance(batch, UserDict):
            # Further input validation is left to the huggingface forward call
            batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
            output = self.model(**batch)  # type: ignore (thirdparty)
        else:
            raise ValueError(
                "Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model"
            )
        return output

    def loss(self, outputs, batch):
        if self.config.use_return_dict:
            loss, logits = outputs["loss"], outputs["logits"]
        else:
            # loss is at index 0 in the output tuple, logits are at index 1
            loss, logits = outputs[:2]
        if self.z_loss == 0.0:
            return loss

        # Add a z_loss to the standard loss
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = batch["labels"].view(-1)
        log_z = torch.logsumexp(logits_flat[labels_flat != _HF_IGNORE_INDEX], dim=1)
        log_z2 = log_z**2
        z_loss = log_z2.mean() * self.z_loss
        if self.config.use_return_dict:
            outputs["loss"] += z_loss
            return outputs["loss"]
        else:
            outputs[0] += z_loss
            return outputs[0]

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        if "generate_output" in batch:
            self.labels = batch.pop("labels")
            return self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=512,
                do_sample=True,
                top_p=0.90,
                top_k=0,
                no_repeat_ngram_size=3,
            )

        return super().eval_forward(batch, outputs)

    def flops_per_batch(self, batch):
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass

        bs, msl = batch["input_ids"].shape[0:2]
        params_flops_per_token = 2 * self.n_active_params
        params_flops_per_seq = params_flops_per_token * msl
        attn_flops_per_seq = self.num_hidden_layers * 2 * 2 * (self.hidden_size * (msl**2))

        # return (24 * 4 * bs * msl * self.num_hidden_layers * (self.hidden_size **2)) * (1. + (msl / (6. * self.hidden_size)) + (self.vocab_size / (16. *  self.num_hidden_layers * self.hidden_size)))

        return (params_flops_per_seq + attn_flops_per_seq) * 3 * bs
