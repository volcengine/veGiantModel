import transformers


def build_tokenizer(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(**args.tokenizer.dict())

    return tokenizer
