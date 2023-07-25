import importlib
import json
import os

import transformers


from veturbollm.config import TaskConfig


def build_tokenizer(args: TaskConfig) -> transformers.PreTrainedTokenizer:
    # parse file to speed up tokenizer loading
    if os.path.isdir(args.tokenizer.pretrained_model_name_or_path):
        with open(os.path.join(args.tokenizer.pretrained_model_name_or_path, "tokenizer_config.json")) as f:
            tokenizer_config = json.load(f)
        tokenizer_class = getattr(transformers, tokenizer_config["tokenizer_class"])
        tokenizer = tokenizer_class.from_pretrained(**args.tokenizer.dict())
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(**args.tokenizer.dict())
    return tokenizer
