import logging
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import load_dataset
from torch.utils.data import ConcatDataset, Dataset, Subset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veturbollm.utils.distribution import main_process_first
from veturbollm.global_vars import get_args

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

logger = logging.getLogger(__name__)


def get_train_eval_dataset(tokenizer: Tokenizer) -> Tuple[Dataset, Dataset]:
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    args = get_args()
    if args.dataset.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset.dataset_name, args.dataset.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset.dataset_name,
                args.dataset.dataset_config_name,
                split=f"train[:{args.dataset.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset.dataset_name,
                args.dataset.dataset_config_name,
                split=f"train[{args.dataset.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.dataset.get("train_file") is not None:
            data_files["train"] = args.dataset.train_file
        if args.dataset.get("validation_file") is not None:
            data_files["validation"] = args.dataset.validation_file
        extension = args.dataset.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = args.dataset.get("no_keep_linebreaks", False)
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.dataset.get('validation_split_percentage', 20)}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.dataset.get('validation_split_percentage', 20)}%:]",
                **dataset_args,
            )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    # rank 0 will preprocess the dataset
    with main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.dataset.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.dataset.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.dataset.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if args.dataset.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.dataset.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.dataset.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.dataset.preprocessing_num_workers,
            load_from_cache_file=not args.dataset.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    return train_dataset, eval_dataset
