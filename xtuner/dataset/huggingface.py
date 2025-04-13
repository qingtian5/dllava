# Copyright (c) OpenMMLab. All rights reserved.
import logging
from functools import partial
import json
import numpy as np
import os
from datetime import timedelta
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.utils.misc import get_object_from_string
from torch import distributed as dist
import torch
from xtuner.registry import BUILDER, MAP_FUNC
from .utils import Packer, encode_fn
from itertools import chain

from PIL import Image
import cv2
from pathlib import Path

def detele_iccfile(image_path):
    img = Image.open(image_path)
    img.info.pop('icc_profile', None)
    img.save(image_path)

def image_map_fn(example):
    if example.get('image', None) is None:
        return True

    if example.get('image', None) is not None and isinstance(example['image'], str):
        image_path = example['image']
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("image is None")
            else:
                detele_iccfile(image_path)  
        except Exception as e:
            try:
                image_pil = Image.open(image_path).convert('RGB')
                output_image = str(Path(image_path).with_suffix(".jpg"))
                image_pil.save(output_image)
                detele_iccfile(output_image)  
                example['image'] = output_image
            except Exception as e:
                return False
        else:
            return True
    
    if example.get('image', None) is not None and isinstance(example['image'], list): 
        for i in range(len(example['image'])):
            image_path = example['image'][i]
            try:
                image = cv2.imread(image_path)
                if image is None:
                    raise Exception("image is None")
                else:
                    detele_iccfile(image_path)  
            except Exception as e:
                try:
                    image_pil = Image.open(image_path).convert('RGB')
                    output_image = str(Path(image_path).with_suffix(".jpg"))
                    image_pil.save(output_image)
                    detele_iccfile(output_image)  
                    example['image'][i] = output_image
                except Exception as e:
                    return False
        return True


def get_lengths(example):
    assert 'input_ids' in example
    return {'length': len(example['input_ids'])}

def process(dataset,
            tokenizer,
            max_length,
            dataset_map_fn=None,
            template_map_fn=None,
            max_dataset_length=None,
            split='train',
            remove_unused_columns=False,
            rename_maps=[],
            shuffle_before_pack=True,
            pack_to_max_length=True,
            input_ids_with_output=True,
            with_image_token=False,
            map_num_proc=32,
            system=''):
    """Post-process the dataset loaded from the Hugging Face Hub, or a local
    dataset.

    Args:
        dataset: The dataset to be post-processed.
        tokenizer: The tokenizer processes some raw text as input and outputs
            an Encoding.
        max_length: Max length of the sequence.
        dataset_map_fn: Map the original dataset format to the one defined
            by xTuner.
        template_map_fn: Add the prompt template to the dataset
        max_dataset_length: If the length of the dataset is too long, we can
            randomly extract `max_dataset_length` from it.
        split: Which split of the data to load.
            If `None`, will return a `dict` with all splits (typically
            `datasets.Split.TRAIN` and `datasets.Split.TEST`).
            If given, will return a single Dataset.
        remove_unused_columns: Whether to remove columns from the dataset
            that are not used during training.
        rename_maps: Rename the column name of the dataset.
        shuffle_before_pack: Whether to shuffle the dataset before
            packing them.
        pack_to_max_length: Whether to pack the dataset to the `max_length `.
            This usually improves gpu utilization and therefore reduces
            training time.
        input_ids_with_output: Whether to put the groundtruth output
            corresponding to the question into the dataset. Typically set
            it to True during training and False during testing.
        with_image_token: Whether to convert DEFAULT_IMAGE_TOKEN to
            IMAGE_TOKEN_INDEX. Typically set it to True during the training
            of VLM.
        map_num_proc: Max number of processes when mapping the dataset.
    """

    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]
    elif isinstance(dataset, dict) or isinstance(
            dataset, Config) or isinstance(dataset, ConfigDict):
        dataset = BUILDER.build(dataset)
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]

    # sample `max_dataset_length` items from the original dataset to
    # save time consumed by map function
    if max_dataset_length is not None:
        max_dataset_length = min(max_dataset_length, len(dataset))
        indices = np.random.choice(
            len(dataset), max_dataset_length, replace=False)
        dataset = dataset.select(indices)

    # Extract the useful data for training from the original dataset.
    if dataset_map_fn is not None:
        if isinstance(dataset_map_fn, str):
            map_fn_obj = MAP_FUNC.get(
                dataset_map_fn) or get_object_from_string(dataset_map_fn)
            if map_fn_obj is not None:
                dataset_map_fn = map_fn_obj
            else:
                raise TypeError('dataset_map_fn must be a function or a '
                                "registered function's string in MAP_FUNC, "
                                f"but got a string of '{dataset_map_fn}'")

        if system != '':
            dataset_map_fn = partial(dataset_map_fn, system=system)

        dataset = dataset.map(dataset_map_fn, num_proc=map_num_proc)

    # Add prompt template, such as <|System|>: xxx <|User|>: xxx <|Bot|>: xxx
    if template_map_fn is not None:
        if isinstance(template_map_fn, dict) or isinstance(
                template_map_fn, Config) or isinstance(template_map_fn,
                                                       ConfigDict):
            template_map_fn = BUILDER.build(template_map_fn)
        dataset = dataset.map(template_map_fn, num_proc=map_num_proc)

    for old, new in rename_maps:
        dataset = dataset.rename_column(old, new)

    # remove unused columns
    if pack_to_max_length and (not remove_unused_columns):
        print_log(
            'We have to remove unused columns if '
            '`pack_to_max_length` is set to True.',
            logger='current',
            level=logging.WARNING)
        remove_unused_columns = True

    # remove invalid data
    print(f"before remove invalid data dataset: {len(dataset)}")
    dataset = dataset.filter(
        lambda example: len(example['conversation']) > 0,
        num_proc=map_num_proc)

    print(f"after remove invalid data dataset: {len(dataset)}")

    # tokenize
    if isinstance(tokenizer, dict) or isinstance(
            tokenizer, Config) or isinstance(tokenizer, ConfigDict):
        tokenizer = BUILDER.build(tokenizer)
    dataset = dataset.map(
        partial(
            encode_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            with_image_token=with_image_token,
            input_ids_with_output=input_ids_with_output),
        remove_columns=list(dataset.column_names)
        if remove_unused_columns else None,
        num_proc=map_num_proc)

    if input_ids_with_output:
        # remove data that does not have the valid labels.
        print(f"before remove data that does not have the valid labels dataset: {len(dataset)}")
        # print(dataset[0])
        dataset = dataset.filter(
            lambda example: any(label >= 0 for label in example['labels']),
            num_proc=map_num_proc)
        print(f"after remove data that does not have the valid labels. dataset: {len(dataset)}")

    # pack to max length
    if pack_to_max_length and split == 'train':
        if shuffle_before_pack:
            dataset = dataset.shuffle()
            dataset = dataset.flatten_indices(num_proc=map_num_proc)
        dataset = dataset.map(
            Packer(max_length), batched=True, num_proc=map_num_proc)

    # add 'length'
    # setattr(dataset, 'length', [len(i['input_ids']) for i in dataset])
    dataset = dataset.map(get_lengths, num_proc=1)
    dataset_lenths = [i['length'] for i in dataset]
    print_log(f"token static in dataset: max num_tokens {max(dataset_lenths)}, min num_tokens: {min(dataset_lenths)}, avg num_tokens: {round(sum(dataset_lenths) / len(dataset_lenths))}", logger='current')

    setattr(dataset, 'length', dataset['length'])

    print_log("return processed hf dataset", logger='current')

    return dataset


def process_hf_dataset(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()):
        return process(*args, **kwargs)
    
    xtuner_dataset_timeout = timedelta(
        minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=180)))
    print_log(
        f'xtuner_dataset_timeout = {xtuner_dataset_timeout}', logger='current')
    # monitored barrier requires gloo process group to perform host-side sync.
    group_gloo = dist.new_group(backend='gloo', timeout=xtuner_dataset_timeout)

    if dist.get_rank() == 0:
        dataset = process(*args, **kwargs)
        objects = [dataset]
    else:
        objects = [None]

    # dist.broadcast_object_list(objects, src=0)

    dist.monitored_barrier(group=group_gloo, timeout=xtuner_dataset_timeout)
    dist.broadcast_object_list(objects, src=0)
    return objects[0]



def process_list(*args, **kwargs):
    import json
    import gc
    from datasets import Dataset as HFDataset
    from datasets import DatasetDict, concatenate_datasets

    data_path = kwargs['data_path']

    json_data = json.load(open(data_path[0]))
    for idx in range(len(json_data)):
        if isinstance(json_data[idx]['id'], int):
            json_data[idx]['id'] = str(json_data[idx]['id'])

    text_data = HFDataset.from_list(json_data)
    print_log(f"text_data has {len(text_data)} records", logger='current')
    del json_data
    gc.collect()
    for dp in data_path[1:]:
        json_data = json.load(open(dp))
        for idx in range(len(json_data)):
            if isinstance(json_data[idx]['id'], int):
                json_data[idx]['id'] = str(json_data[idx]['id'])
        text_data = concatenate_datasets([text_data, HFDataset.from_list(json_data)])
        print_log(f"text_data has {len(text_data)} records", logger='current')
        del json_data
        gc.collect()

    return text_data

def process_list_dataset(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()):
        return process_list(*args, **kwargs)
    
    xtuner_dataset_timeout = timedelta(
        minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=60)))
    print_log(
        f'xtuner_dataset_timeout = {xtuner_dataset_timeout}', logger='current')
    # monitored barrier requires gloo process group to perform host-side sync.
    group_gloo = dist.new_group(backend='gloo', timeout=xtuner_dataset_timeout)

    if dist.get_rank() == 0:
        # dataset = process(*args, **kwargs)
        text_data = process_list(*args, **kwargs)
        objects = [text_data]
    else:
        objects = [None]

    # dist.broadcast_object_list(objects, src=0)

    dist.monitored_barrier(group=group_gloo, timeout=xtuner_dataset_timeout)
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


def process_sharding_files(data_path, sampler_name, group_batch_size, tokenizer, max_length, dataset_map_fn, template_map_fn, skip_filer_and_map, seed, work_dir):
    
    rank, world_size = dist.get_rank(), dist.get_world_size()

    file_list = os.listdir(data_path)
    file_list.sort()

    sharding_indices_file_path = os.path.join(work_dir, data_path.replace('/', '_') + '_' + sampler_name.lower() + '_g' + str(group_batch_size) + '_s' + str(int(skip_filer_and_map)) + '_w' + str(world_size) + '.indices')
    
    print_log(f'rank-{rank}, world_size-{world_size}, process_sharding_files is: {sharding_indices_file_path}', logger='current')

    if not os.path.exists(sharding_indices_file_path):
        print(f'rank-{rank}, world_size-{world_size}, start to build sharding_indices_file_path: {sharding_indices_file_path}\n')
        build_sharding_indices(data_path=data_path, sampler_name=sampler_name, group_batch_size=group_batch_size, tokenizer=tokenizer, max_length=max_length, 
                               dataset_map_fn=dataset_map_fn, template_map_fn=template_map_fn, skip_filer_and_map=skip_filer_and_map, 
                               seed=seed, sharding_indices_file_path=sharding_indices_file_path, rank=rank, world_size=world_size, file_list=file_list)
    else:
        print_log(f'rank-{rank}, world_size-{world_size}, skip building sharding_indices_file_path: {sharding_indices_file_path}', logger='current')

    dist.barrier()

    print(f'rank-{rank}, world_size-{world_size}, start to load sharding_indices_file_path: {sharding_indices_file_path}\n')
    with open(sharding_indices_file_path, "r")as f:
        indices_mapping = json.load(f)

    data_on_rank = []
    idx = 0
    for i in range(len(file_list)):
        file_path = os.path.join(data_path, file_list[i])
        with open(file_path) as f:
            json_data = json.load(f)
            kk = 0
            for example in json_data:
                valid, to_rank, to_index = indices_mapping[idx]
                # if rank == 0:
                #     print(f'rank-{rank}, world_size-{world_size}, loaded item, valid={valid}, to_rank={to_rank}, to_index={to_index}\n')
                if valid == 1 and rank == to_rank:
                    # fix type error
                    if isinstance(example['id'], int):
                        example['id'] = str(example['id'])
                    data_on_rank += [(to_index, example)]
                idx += 1
                kk += 1

    indexed_data_on_rank = [None] * len(data_on_rank)
    for i, example in data_on_rank:
        indexed_data_on_rank[i] = example
    
    if len(indexed_data_on_rank) > 0:
        if 'image' not in indexed_data_on_rank[0]:
            indexed_data_on_rank[0]['image'] = None

    dataset = HFDataset.from_list(indexed_data_on_rank)

    if not skip_filer_and_map:
        # Extract the useful data for training from the original dataset.
        if dataset_map_fn is not None:
            if isinstance(dataset_map_fn, str):
                map_fn_obj = MAP_FUNC.get(dataset_map_fn) or get_object_from_string(dataset_map_fn)
                if map_fn_obj is not None:
                    dataset_map_fn = map_fn_obj
                else:
                    raise TypeError('dataset_map_fn must be a function or a '
                                    "registered function's string in MAP_FUNC, "
                                    f"but got a string of '{dataset_map_fn}'")

            dataset = dataset.map(dataset_map_fn, num_proc=32)

        # Add prompt template, such as <|System|>: xxx <|User|>: xxx <|Bot|>: xxx
        if template_map_fn is not None:
            if isinstance(template_map_fn, dict) or isinstance(
                    template_map_fn, Config) or isinstance(template_map_fn,
                                                        ConfigDict):
                template_map_fn = BUILDER.build(template_map_fn)
            dataset = dataset.map(template_map_fn, num_proc=32)


        # tokenize
        if isinstance(tokenizer, dict) or isinstance(
                tokenizer, Config) or isinstance(tokenizer, ConfigDict):
            tokenizer = BUILDER.build(tokenizer)
        dataset = dataset.map(
            partial(
                encode_fn,
                tokenizer=tokenizer,
                max_length=max_length,
                with_image_token=True,
                input_ids_with_output=True),
            remove_columns=None,
            num_proc=32)

        # add 'length'
        # setattr(dataset, 'length', [len(i['input_ids']) for i in dataset])
        dataset = dataset.map(get_lengths, num_proc=1)
        setattr(dataset, 'length', dataset['length'])

    print(f'rank-{rank}, world_size-{world_size}, built a sharded dataset, length is: {len(dataset)}\n')
    return dataset
    

def build_sharding_indices(data_path, sampler_name, group_batch_size, tokenizer, max_length, dataset_map_fn, template_map_fn, skip_filer_and_map, seed, sharding_indices_file_path, rank, world_size, file_list):
    tagged_indices_flattened = [] # 1 for ok, 0 for invalid
    length_list_all = []
    for i in range(len(file_list)):
        file_path = os.path.join(data_path, file_list[i])
        cur_dataset = load_dataset_from_json(file_path)
        if i % world_size == rank:
            file_size, valid_size, tagged_indices, length_list, tokens_list = post_process_sharding(dataset=cur_dataset,
                                                                                                    tokenizer=tokenizer,
                                                                                                    max_length=max_length,
                                                                                                    dataset_map_fn=dataset_map_fn,
                                                                                                    template_map_fn=template_map_fn,
                                                                                                    with_image_token=True,
                                                                                                    skip_filer_and_map=skip_filer_and_map)
            print(f'rank-{rank}, world_size-{world_size}, processing: {file_path}, length = {file_size}, valid_size = {valid_size}')
            # print(f'rank-{rank}, world_size-{world_size}, tagged_indices: {tagged_indices}')
            # print(f'rank-{rank}, world_size-{world_size}, length_list: {length_list}')

            print_log(f"token static in dataset: max num_tokens {max(tokens_list)}, min num_tokens: {min(tokens_list)}, avg num_tokens: {round(sum(tokens_list) / len(tokens_list))}", logger='current')

            tagged_indices_flattened.extend(tagged_indices)
            length_list_all.extend(length_list)
        else:
            tagged_indices_flattened.extend([0 for _ in range(len(cur_dataset))])
            length_list_all.extend([0 for _ in range(len(cur_dataset))])

    total_size = len(tagged_indices_flattened)

    valid_list_tensor = torch.tensor(tagged_indices_flattened, device=torch.cuda.current_device())
    dist.all_reduce(valid_list_tensor, op=dist.ReduceOp.SUM)
    tagged_indices_flattened_gathered = valid_list_tensor.tolist()
    valid_size = torch.sum(torch.eq(valid_list_tensor, 1).int())

    if sampler_name == "LengthGroupedSampler":
        assert skip_filer_and_map == False
        assert group_batch_size > 0

        length_list_tensor = torch.tensor(length_list_all, device=torch.cuda.current_device())
        dist.all_reduce(length_list_tensor, op=dist.ReduceOp.SUM)
        length_list_all_gathered = length_list_tensor.tolist()

    if rank == 0:
        if sampler_name == "LengthGroupedSampler":
            assert skip_filer_and_map == False
            assert group_batch_size > 0

            num_samples = valid_size // world_size * world_size # will ignore the last world_size - 1 items at most

            group_batch_size = group_batch_size * 50

            print(f'rank-{rank}, world_size-{world_size}, using LengthGroupedSampler, num_samples={num_samples}, group_batch_size={group_batch_size}\n')

            valid_length_list_all_gathered = [0] * valid_size
            valid_cnt = 0
            for i in range(total_size):
                if tagged_indices_flattened_gathered[i] == 1:
                    valid_length_list_all_gathered[valid_cnt] = length_list_all_gathered[i]
                    valid_cnt += 1

            g = torch.Generator()
            g.manual_seed(seed)
            random_indices = torch.randperm(num_samples, generator=g).tolist() # j=random_indice[i], j is the index
            rank_size = num_samples // world_size # total size for each rank
            random_indices_list_by_rank = [random_indices[i:i + rank_size] for i in range(0, num_samples, rank_size)]
            
            valid_indices_mapping = [(0, -1, -1)] * total_size
            for rank_id, rank_random_indice in enumerate(random_indices_list_by_rank):
                sorted_indice_list = []
                for k in range(0, len(rank_random_indice), group_batch_size): # for each group
                    current_group_indice = rank_random_indice[k:k+group_batch_size]
                    sorted_indice_list += [sorted(current_group_indice, key=lambda x: valid_length_list_all_gathered[x], reverse=True)]
                sorted_indice_flattened = list(chain.from_iterable(sorted_indice_list))
                for i in range(len(sorted_indice_flattened)):
                    from_index = sorted_indice_flattened[i]
                    valid_indices_mapping[from_index] = (1, rank_id, i)

            # map to entire dataset
            indices_mapping = [(-1, -1, -1)] * total_size
            valid_cnt = 0
            for k in range(total_size):
                if k % 10000 == 0:
                    dist.barrier()
                    print_log(f'rank-{rank}, world_size-{world_size}, status: {k}/{total_size}', logger='current')
                if tagged_indices_flattened_gathered[k] == 1:
                    if valid_cnt < num_samples:
                        indices_mapping[k] = valid_indices_mapping[valid_cnt]
                    else:
                        indices_mapping[k] = (1, -1, -1)
                    valid_cnt += 1
                else:
                    indices_mapping[k] = (0, -1, -1)

            with open(sharding_indices_file_path, "w")as f:
                json.dump(indices_mapping, f)
            

        else: 
            # DefaultSampler
            num_samples = valid_size // world_size * world_size # will ignore the last world_size - 1 items at most

            print(f'rank-{rank}, world_size-{world_size}, valid_size = {valid_size}, num_samples: {num_samples} in DefaultSampler')

            g = torch.Generator()
            g.manual_seed(seed)
            random_indices = torch.randperm(num_samples, generator=g).tolist() # j=random_indice[i], j is the index
            rank_size = num_samples // world_size # total size for each rank
            random_indices_list_by_rank = [random_indices[i:i + rank_size] for i in range(0, num_samples, rank_size)]
            
            print(f"valid indices mapping")

            valid_indices_mapping = [(1, -1, -1)] * num_samples
            for rank_id, rank_random_indice in enumerate(random_indices_list_by_rank):
                for i in range(len(rank_random_indice)):
                    from_index = rank_random_indice[i]
                    valid_indices_mapping[from_index] = (1, rank_id, i)

            # map to entire dataset
            print(f"map to entire dataset")
            indices_mapping = [(-1, -1, -1)] * total_size
            valid_cnt = 0
            for k in range(total_size):
                if k % 10000 == 0:
                    dist.barrier()
                    print_log(f'rank-{rank}, world_size-{world_size}, status: {k}/{total_size}', logger='current')
                if tagged_indices_flattened_gathered[k] == 1:
                    if valid_cnt < num_samples:
                        indices_mapping[k] = valid_indices_mapping[valid_cnt]
                    else:
                        indices_mapping[k] = (1, -1, -1)
                    valid_cnt += 1
                else:
                    indices_mapping[k] = (0, -1, -1)

            print(f"rank-{rank}, world_size-{world_size}, sharding_indices_file_path: {sharding_indices_file_path}")

            with open(sharding_indices_file_path, "w")as f:
                json.dump(indices_mapping, f)

            print(f"rank-{rank}, world_size-{world_size}, {sharding_indices_file_path} dump success")
    else:
        for k in range(total_size):
            if k % 10000 == 0:
                dist.barrier()
                print(f'rank-{rank}, world_size-{world_size}, status: {k}/{total_size}')

def load_dataset_from_json(file_path):
    json_data = json.load(open(file_path))
    for idx in range(len(json_data)):
        if isinstance(json_data[idx]['id'], int):
            json_data[idx]['id'] = str(json_data[idx]['id'])
    if len(json_data) > 0:
        if 'image' not in json_data[0]:
            json_data[0]['image'] = None
    dataset = HFDataset.from_list(json_data)
    return dataset


def post_process_sharding(dataset,
                        tokenizer,
                        max_length,
                        dataset_map_fn=None,
                        template_map_fn=None,
                        input_ids_with_output=True,
                        with_image_token=False,
                        map_num_proc=32,
                        skip_filer_and_map=False):

    file_size, valid_size = len(dataset), 0
    tagged_indices, length_list, tokens_list = [1] * file_size, [0] * file_size, [0] * file_size

    if skip_filer_and_map:
        return file_size, file_size, tagged_indices, None

    # Extract the useful data for training from the original dataset.
    if dataset_map_fn is not None:
        dataset = dataset.map(dataset_map_fn, num_proc=map_num_proc)

    # Add prompt template, such as <|System|>: xxx <|User|>: xxx <|Bot|>: xxx
    if template_map_fn is not None:
        if isinstance(template_map_fn, dict) or isinstance(template_map_fn, Config) or isinstance(template_map_fn, ConfigDict):
            template_map_fn = BUILDER.build(template_map_fn)
        dataset = dataset.map(template_map_fn, num_proc=map_num_proc)

    # tokenize
    if isinstance(tokenizer, dict) or isinstance(tokenizer, Config) or isinstance(tokenizer, ConfigDict):
        tokenizer = BUILDER.build(tokenizer)

    # filtering
    for i, example in enumerate(dataset):
        if len(example['conversation']) > 0:
            encoded_ex = encode_fn(example=example, tokenizer=tokenizer, max_length=max_length, with_image_token=with_image_token, input_ids_with_output=input_ids_with_output)
            # image_valid = image_map_fn(example)
            if any(label >= 0 for label in encoded_ex['labels']):
                if 'input_ids' in encoded_ex:
                    if example.get('image', None) is not None and isinstance(example['image'], list) and len(example['image']) > 1: 
                        if len(example['image']) != encoded_ex['input_ids'].count(-200):
                            continue
                        tokens_list[i] = len(encoded_ex['input_ids'])
                        length_list[i] = 200
                    elif example.get('image', None) is not None and isinstance(example['image'], list) and len(example['image']) == 1: 
                        tokens_list[i] = len(encoded_ex['input_ids'])
                        length_list[i] = 100
                    elif example.get('image', None) is not None and isinstance(example['image'], str):
                        tokens_list[i] = len(encoded_ex['input_ids'])
                        length_list[i] = 100
                    elif example.get('image', None) is None: 
                        tokens_list[i] = len(encoded_ex['input_ids'])
                        length_list[i] = -100
                valid_size += 1
            else:
                tagged_indices[i] = 0
        else:
            tagged_indices[i] = 0

    return file_size, valid_size, tagged_indices, length_list, tokens_list
