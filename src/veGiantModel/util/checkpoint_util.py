import torch
import os
import sys
from megatron import print_rank_0
from megatron.checkpointing import get_checkpoint_tracker_filename, get_checkpoint_name

def get_ckpt_name(load_dir):
    if not os.path.exists(load_dir):
        return load_dir

    tracker_filename = get_checkpoint_tracker_filename(load_dir)
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        print_rank_0('    will not load checkpoints and will start from '
                        'random')
        return load_dir
    else:
        iteration = 0
        with open(tracker_filename, 'r') as f:
            metastring = f.read().strip()
            try:
                iteration = int(metastring)
            except ValueError:
                release = metastring == 'release'
                if not release:
                    print('ERROR: Invalid metadata file {}. Exiting'.format(
                        tracker_filename))
                    sys.exit()
        return get_checkpoint_name(load_dir, iteration)


def build_pipe_name_map(num_layers):
    pipe_name_maps = {}
    pipe_name_maps['tied_modules.SharedEmbedding.embedding_weight'] = 'word_embeddings.weight'
    pipe_name_maps['tied_modules.SharedEmbedding.position_embeddings.weight'] = 'position_embeddings.weight'
    for i in (range(num_layers)):
        pipe_idx = i + 2
        pipe_name_maps[f'{pipe_idx}.input_layernorm.weight'] = f'layers.{i}.input_layernorm.weight'
        pipe_name_maps[f'{pipe_idx}.input_layernorm.bias'] = f'layers.{i}.input_layernorm.bias'
        pipe_name_maps[f'{pipe_idx}.attention.query_key_value.weight'] = f'layers.{i}.self_attention.query_key_value.weight'
        pipe_name_maps[f'{pipe_idx}.attention.query_key_value.bias'] = f'layers.{i}.self_attention.query_key_value.bias'
        pipe_name_maps[f'{pipe_idx}.attention.dense.weight'] = f'layers.{i}.self_attention.dense.weight'
        pipe_name_maps[f'{pipe_idx}.attention.dense.bias'] = f'layers.{i}.self_attention.dense.bias'
        pipe_name_maps[f'{pipe_idx}.post_attention_layernorm.weight'] = f'layers.{i}.post_attention_layernorm.weight'
        pipe_name_maps[f'{pipe_idx}.post_attention_layernorm.bias'] = f'layers.{i}.post_attention_layernorm.bias'

        pipe_name_maps[f'{pipe_idx}.mlp.dense_h_to_4h.weight'] = f'layers.{i}.mlp.dense_h_to_4h.weight'
        pipe_name_maps[f'{pipe_idx}.mlp.dense_h_to_4h.bias'] = f'layers.{i}.mlp.dense_h_to_4h.bias'
        pipe_name_maps[f'{pipe_idx}.mlp.dense_4h_to_h.weight'] = f'layers.{i}.mlp.dense_4h_to_h.weight'
        pipe_name_maps[f'{pipe_idx}.mlp.dense_4h_to_h.bias'] = f'layers.{i}.mlp.dense_4h_to_h.bias'

    pipe_name_maps[f'{num_layers+3}.weight'] = 'final_layernorm.weight'
    pipe_name_maps[f'{num_layers+3}.bias'] = 'final_layernorm.bias'
    return pipe_name_maps

def load_megatron_model_state(module, num_layers, checkpoint, load_optimizer, load_module_strict=True):
    loaded_model = checkpoint['model']
    pipe_name_maps = build_pipe_name_map(num_layers)
    module_param_names = []
    for name, param in module.named_parameters():
        module_param_names.append(name)

    loaded_dict = {}
    language_model = loaded_model['language_model']
    embedding = language_model['embedding']
    transformer = language_model['encoder']

    param_idx = 0
    param_indices_map = {}
    wd_param_indices = []
    no_wd_param_indices = []

    for name, param in embedding.items():
        for sub_name, sub_param in param.items():
            key = name + '.' + sub_name
            loaded_dict[key] = sub_param
            param_indices_map[key] = param_idx
            param_idx += 1


    for name, param in transformer.items():
        loaded_dict[name] = param
        param_idx += 1
        param_indices_map[name] = param_idx


    new_loaded_dict = {}
    import megatron
    args = megatron.get_args()

    for name in module_param_names:
        load_name = pipe_name_maps[name]
        new_loaded_dict[name] = loaded_dict[load_name]

        # we track the index of any relevant parameter in the current stage
        # and look for its original index in the Megatron checkpoint.
        # this helps help point this parameter back to its original state
        if load_optimizer:
            if name in args.weight_decay_names:
                wd_param_indices.append(param_indices_map[load_name])
            elif name in args.no_weight_decay_names:
                no_wd_param_indices.append(param_indices_map[load_name])
            else:
                assert False, name

    new_loaded_dict['tied_modules.SharedEmbedding.word_embeddings.weight'] = new_loaded_dict['tied_modules.SharedEmbedding.embedding_weight']
    module.load_state_dict(new_loaded_dict, load_module_strict)

    return wd_param_indices, no_wd_param_indices

def test():
    checkpoint_name = '/opt/tiger/chkpt-data/iter_0250000/mp_rank_00/model_optim_rng.pt'
    # checkpoint_name = '/opt/tiger/ds-chkpt-data/global_step0/mp_rank_03_model_states.pt'
    state_dict = torch.load(checkpoint_name, map_location='cpu')
    print(f'load checkpont from {checkpoint_name}', flush=True)
    value = state_dict.keys()
    print(f'checkpoint keys: {value}')
    value = state_dict['model'].keys()
    print(f'checkpoint[model] keys: {value}')
    value = state_dict['model']['language_model'].keys()
    print(f'checkpoint[model][language_model] keys: {value}')
    value = state_dict['model']['language_model']['embedding'].keys()
    print(f'checkpoint[model][language_model][embedding] keys: {value}')
    value = state_dict['model']['language_model']['encoder'].keys()
    print(f'checkpoint[model][language_model][encoder] keys: start')
    for k in value:
        print(f'{k}')
    print(f'checkpoint[model][language_model][encoder] keys: finish')
    pass

if __name__ == '__main__':
    test()
