# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT2"""
import torch
import os
import numpy as np
import time
import sys

_cwd = os.path.dirname(os.path.abspath(__file__))
_giantModel_dir = os.path.join(_cwd, '../../src')
sys.path.append(_giantModel_dir)

from initialize import initialize_megatron, initialize_pipeline
from gpt_piped import GPTModelPiped

from megatron import get_args, mpu
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import print_rank_0
from megatron.learning_rates import AnnealingLR
from megatron.training import build_train_valid_test_data_iterators
from megatron.data.gpt2_dataset import get_indexed_dataset_, get_train_valid_test_split_, _num_tokens, _num_epochs, _build_doc_idx, _build_shuffle_idx
from deepspeed.utils import log_dist

def _build_index_mappings(name, data_prefix, documents, sizes,
                        num_samples, seq_length, seed):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
    training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    log_dist(f' >>>> Entering _build_index_mappings', ranks=[-1])
    # Number of tokens in each epoch and number of required epochs.
    args = get_args()
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_{}_indexmap'.format(args.rank, name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}sl'.format(seq_length)
    _filename += '_{}s'.format(seed)
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    device_count = torch.cuda.device_count()
    if (not os.path.isfile(doc_idx_filename)) or \
    (not os.path.isfile(sample_idx_filename)) or \
    (not os.path.isfile(shuffle_idx_filename)):

        log_dist(f' > WARNING: could not find index map files, building '
                    'the indices ...', ranks=[-1])
        # doc-idx.
        start_time = time.time()
        doc_idx = _build_doc_idx(documents, num_epochs, np_rng)
        np.save(doc_idx_filename, doc_idx, allow_pickle=True)
        log_dist(' > elasped time to build and save doc-idx mapping '
                    '(seconds): {:4f}'.format(time.time() - start_time), ranks=[-1])
        # sample-idx.
        start_time = time.time()
        # Use C++ implementation for speed.
        # First compile and then import.
        from megatron.data.dataset_utils import compile_helper
        compile_helper()
        from megatron.data import helpers
        assert doc_idx.dtype == np.int32
        assert sizes.dtype == np.int32
        sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length,
                                            num_epochs, tokens_per_epoch)
        # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
        #                               num_epochs, tokens_per_epoch)
        np.save(sample_idx_filename, sample_idx, allow_pickle=True)
        log_dist(' > elasped time to build and save sample-idx mapping '
                    '(seconds): {:4f}'.format(time.time() - start_time), ranks=[-1])
        # shuffle-idx.
        start_time = time.time()
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        shuffle_idx = _build_shuffle_idx(sample_idx.shape[0] - 1, np_rng)
        np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
        log_dist(' > elasped time to build and save shuffle-idx mapping'
                    ' (seconds): {:4f}'.format(time.time() - start_time), ranks=[-1])

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    assert counts[0].item() == torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())

    # Load mappings.
    start_time = time.time()
    log_dist(' > loading doc-idx mapping from {}'.format(
        doc_idx_filename))

    if not os.path.isfile(doc_idx_filename):
        log_dist(' > loading doc-idx mapping from {} failed, file not exist'.format(
        doc_idx_filename), ranks=[-1])

    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
    log_dist(' > loading sample-idx mapping from {}'.format(
        sample_idx_filename), ranks=[-1])
    if not os.path.isfile(sample_idx_filename):
        log_dist(' > loading doc-idx mapping from {} failed, file not exist'.format(
        sample_idx_filename), ranks=[-1])
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    log_dist(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename), ranks=[-1])
    if not os.path.isfile(shuffle_idx_filename):
        log_dist(' > loading doc-idx mapping from {} failed, file not exist'.format(
        shuffle_idx_filename), ranks=[-1])
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    log_dist('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time), ranks=[-1])
    log_dist('    total number of samples: {}'.format(
        sample_idx.shape[0]), ranks=[-1])
    log_dist('    total number of epochs: {}'.format(num_epochs), ranks=[-1])

    log_dist(f' >>>> exiting _build_index_mappings', ranks=[-1])
    return doc_idx, sample_idx, shuffle_idx
    
class GPT2DatasetFixed(torch.utils.data.Dataset):
    def __init__(self, name, data_prefix, documents, indexed_dataset,
                 num_samples, seq_length, seed):

        self.name = name
        self.indexed_dataset = indexed_dataset

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            self.name, data_prefix, documents, self.indexed_dataset.sizes,
            num_samples, seq_length, seed)

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                                    offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(self.indexed_dataset.get(
                self.doc_idx[doc_index_l],
                length=offset_l + 1))
            sample = np.concatenate(sample_list)

        return {'text': np.array(sample, dtype=np.int64)}



def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                    train_valid_test_num_samples,
                                    seq_length, seed, skip_warmup):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = GPT2DatasetFixed(name, data_prefix,
                                  documents, indexed_dataset,
                                  train_valid_test_num_samples[index],
                                  seq_length, seed)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)

def model_provider():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPTModelPiped()
    return model

def lr_scheduler_builder(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = args.warmup * num_iters
  
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=warmup_iter,
        total_iters=num_iters,
        decay_style=args.lr_decay_style,
        last_iter=init_step,
        min_lr=args.min_lr,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)
    
    return lr_scheduler


def pretrain(model_provider, args_defaults={}):
    initialize_megatron(args_defaults=args_defaults)
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model and optimizer').start()
    model = model_provider()
    engine, optimizer, lr_scheduler = initialize_pipeline(model, None, None, lr_scheduler_builder)
    timers('model and optimizer').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    print_rank_0('training ...')

    train(engine, optimizer, lr_scheduler)

def traing_log(loss_dict, iteration):
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)
    add_to_logging('forward')
    add_to_logging('backward')
    add_to_logging('backward-backward')
    add_to_logging('backward-allreduce')
    add_to_logging('backward-master-grad')
    add_to_logging('backward-clip-grad')
    add_to_logging('optimizer')
    add_to_logging('batch generator')

    if writer and torch.distributed.get_rank() == 0:
        writer.add_scalar('loss', loss_dict, iteration)
        normalizer = iteration % args.log_interval
        if normalizer == 0:
            normalizer = args.log_interval
        timers.write(timers_to_log, writer, iteration,
                     normalizer=normalizer)

def train_valid_test_dataset_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

def train(engine, optimizer, lr_scheduler):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Turn on training mode which enables dropout.
    engine.train()

    # Iterations.
    iteration = args.iteration

    timers('interval time').start()

    train_data_iterator, valid_data_iterator, test_data_iterator \
        = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)

    log_dist(f' >>>> start training', ranks=[-1])
    while iteration < args.train_iters:
        engine.train_batch(train_data_iterator)

if __name__ == "__main__":
    pretrain(model_provider,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
