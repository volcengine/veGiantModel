import logging
from typing import Tuple, Union

import torch
import copy
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veturbollm.global_vars import get_args
from veturbollm.utils.distribution import main_process_first

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
logger = logging.getLogger(__name__)


def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    # print(strings)
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=2048,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def get_sft_train_dataset(tokenizer: Tokenizer) -> Dataset:
    args = get_args()
    data_files = {}
    data_files["train"] = args.dataset.train_file
    extension = args.dataset.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None or tokenizer.pad_token == "":
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None or tokenizer.eos_token == "":
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None or tokenizer.bos_token == "":
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None or tokenizer.unk_token == "":
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    tokenizer.add_special_tokens(special_tokens_dict)

    def pre_tokenize_function(examples):
        keys = examples.keys()
        data_list = []
        for i in range(len(examples["input"])):
            data_list.append({k: examples[k][i] for k in keys})

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example["input"] != "" else prompt_no_input.format_map(example)
            for example in data_list
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in data_list]

        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = _HF_IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    # rank 0 will preprocess the dataset
    with main_process_first():
        tokenized_datasets = raw_datasets.map(
            pre_tokenize_function,
            batched=True,
            num_proc=8,
            remove_columns=["input", "output", "instruction"],
            desc="Running tokenizer on dataset",
        )

    def preprocess_texts(examples):
        # print(examples)
        input_ids, labels = tuple(
            [torch.tensor(example) for example in examples[key]] for key in ("input_ids", "labels")
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
        )

    with main_process_first():
        lm_datasets = tokenized_datasets.map(
            preprocess_texts,
            batched=True,
            num_proc=8,
        )

    return lm_datasets["train"]
