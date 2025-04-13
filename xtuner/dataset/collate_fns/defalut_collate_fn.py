# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist

from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX

from mmengine.config import Config, ConfigDict
from xtuner.registry import BUILDER, MAP_FUNC
from transformers import AutoTokenizer

from mmengine import print_log

llm_name_or_path = '/mnt/pfs-guan-ssai/cv/moxu/MindGPT-V6-16B-stage2-26000/'
max_length = int(4096 - 5 * (336 / 14)**2)

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

if isinstance(tokenizer, dict) or isinstance(
        tokenizer, Config) or isinstance(tokenizer, ConfigDict):
    tokenizer = BUILDER.build(tokenizer)

from ..map_fns import llava_map_fn, template_map_fn_factory
from mmengine.utils.misc import get_object_from_string
dataset_map_fn=llava_map_fn
if isinstance(dataset_map_fn, str):
    map_fn_obj = MAP_FUNC.get(
        dataset_map_fn) or get_object_from_string(dataset_map_fn)
    if map_fn_obj is not None:
        dataset_map_fn = map_fn_obj
    else:
        raise TypeError('dataset_map_fn must be a function or a '
                        "registered function's string in MAP_FUNC, "
                        f"but got a string of '{dataset_map_fn}'")

from xtuner.utils import PROMPT_TEMPLATE
prompt_template = PROMPT_TEMPLATE.internlm2_chat
template_map_fn=dict(
    type=template_map_fn_factory, template=prompt_template)
if template_map_fn is not None:
    if isinstance(template_map_fn, dict) or isinstance(
            template_map_fn, Config) or isinstance(template_map_fn,
                                                    ConfigDict):
        template_map_fn = BUILDER.build(template_map_fn)

def default_collate_fn(
        instances: Sequence[Dict],
        pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
        return_hf_format: bool = False) -> Dict[str, torch.Tensor]:

    input_ids = []
    labels = []

    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    if has_image:
        pixel_values = []
        dino_pixel_values = []
        convnext_pixel_values = []
        num_pixel_values = []

    if 'input_ids' not in instances[0]:

        from ..utils import encode_fn

        for example in instances:
            dataset_map_results = dataset_map_fn(example)
            example.update(dataset_map_results)

        for example in instances:
            template_map_results = template_map_fn(example)
            example.update(template_map_results)
        
        for example in instances:
            encode_results = encode_fn(example, tokenizer=tokenizer, max_length = max_length, with_image_token=True, input_ids_with_output=True)
            example.update(encode_results)


    num_instance = len(instances)

    instances = list(filter(
        lambda example: any(label >= 0 for label in example['labels']),
        instances
    ))

    while len(instances) < num_instance:
        instances += [instances[-1]]
        print("pad sample from tail")

    # print_log(f"default_collate_fn ids: {[example['id'] for example in instances]}", logger='current')

    for example in instances:

        input_ids.append(torch.tensor(example['input_ids']))
        labels.append(torch.tensor(example['labels']))
        if has_image:
            # print(f"example['pixel_values']: {example['pixel_values'].shape}")
            pixel_values.append(example['pixel_values'])

            if example.get('dino_pixel_values', None) is not None:
                dino_pixel_values.append(example['dino_pixel_values'])

            if example.get('convnext_pixel_values', None) is not None:
                convnext_pixel_values.append(example['convnext_pixel_values'])

            if example.get('num_pixel_values', None) is not None:
                num_pixel_values.append(example['num_pixel_values'])

    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    if len(num_pixel_values) > 0:
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(pad_index),
            'labels': labels, 
            'num_pixel_values': num_pixel_values
        }
    else:
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(pad_index),
            'labels': labels
        }

    if has_image:
        # pixel_values = torch.stack(pixel_values)
        
        pixel_values = torch.cat(pixel_values, dim=0)
        data_dict['pixel_values'] = pixel_values

        if len(dino_pixel_values) > 0:
            dino_pixel_values = torch.cat(dino_pixel_values, dim=0)
            data_dict['dino_pixel_values'] = dino_pixel_values

        if len(convnext_pixel_values) > 0:
            convnext_pixel_values = torch.cat(convnext_pixel_values, dim=0)
            data_dict['convnext_pixel_values'] = convnext_pixel_values

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}

def dynamicimages_collate_fn(
        instances: Sequence[Dict],
        pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
        return_hf_format: bool = False) -> Dict[str, torch.Tensor]:

    input_ids = []
    labels = []
    data_ids = []

    has_image_thumbnail = any(inst.get('pixel_values_thumbnail') is not None for inst in instances)
    has_image_tiles = any(inst.get('pixel_values_tiles') is not None for inst in instances)

    has_image = has_image_thumbnail or has_image_tiles

    if has_image:
        pixel_values_thumbnail = []
        pixel_values_tiles = []
        convnext_pixel_values_thumbnail = []
        convnext_pixel_values_tiles = []
        num_pixel_values = []

    # print_log(f"default_collate_fn ids: {[example['id'] for example in instances]}", logger='current')

    for example in instances:

        data_ids.append(example['id'])

        input_ids.append(torch.tensor(example['input_ids']))
        labels.append(torch.tensor(example['labels']))
        if has_image:
            # print(f"example['pixel_values']: {example['pixel_values'].shape}")

            if example['pixel_values_thumbnail'] is not None:
                if isinstance(example['pixel_values_thumbnail'], list):
                    pixel_values_thumbnail.extend(example['pixel_values_thumbnail'])
                else:
                    pixel_values_thumbnail.append(example['pixel_values_thumbnail'])
            if example['pixel_values_tiles'] is not None:
                if isinstance(example['pixel_values_tiles'], list):
                    pixel_values_tiles.extend(example['pixel_values_tiles'])
                else:
                    pixel_values_tiles.append(example['pixel_values_tiles'])

            if example.get('convnext_pixel_values_thumbnail', None) is not None:
                convnext_pixel_values_thumbnail.append(example['convnext_pixel_values_thumbnail'])
            if example.get('convnext_pixel_values_tiles', None) is not None:
                convnext_pixel_values_tiles.append(example['convnext_pixel_values_tiles'])

            if isinstance(example['num_pixel_values'], list):
                num_pixel_values.extend(example['num_pixel_values'])
            else:
                num_pixel_values.append(example['num_pixel_values'])

    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)


    data_dict = {
        'input_ids': input_ids,
        'attention_mask': input_ids.ne(pad_index),
        'labels': labels,
        'data_ids': data_ids
    }

    if has_image:
        # pixel_values = torch.stack(pixel_values)
        
        if len(pixel_values_thumbnail) > 0:
            max_resolution = -1
            image_size = None
            for x in pixel_values_thumbnail:
                # print(x.shape)
                if x.shape[2] * x.shape[3] > max_resolution:
                    max_resolution = x.shape[2] * x.shape[3]
                    image_size = (x.shape[2], x.shape[3])
            # print(f"image_size: {image_size}")

            padded_pixel_values_thumbnail = []
            for x in pixel_values_thumbnail:
                if x.shape[2] != image_size[0] and x.shape[3] != image_size[1]:
                    padded_pixel_values_thumbnail.append(
                        torch.nn.functional.pad(x, (0, image_size[1] - x.shape[3], 0, image_size[0] - x.shape[2]), mode="constant", value=0)
                    )
                else:
                    padded_pixel_values_thumbnail.append(x)

            padded_pixel_values_thumbnail = torch.cat(padded_pixel_values_thumbnail, dim=0)
            data_dict['pixel_values'] = padded_pixel_values_thumbnail
            data_dict['image_size'] = image_size

        if len(convnext_pixel_values_thumbnail):
            data_dict['convnext_pixel_values'] = torch.cat(convnext_pixel_values_thumbnail, dim=0)

        if len(pixel_values_tiles) > 0:
            
            if len(pixel_values_thumbnail) > 0:
                max_tiles = max(num_pixel_values) - 1
                max_tiles = max(max_tiles, 1)
            else:
                max_tiles = max(num_pixel_values)

            padded_pixel_values_tiles = []

            for x in pixel_values_tiles:
                if x.shape[0] != max_tiles:
                    num_padding = max_tiles - x.shape[0]
                    x = torch.cat([x, torch.zeros(num_padding, 3, x.shape[2], x.shape[3])], dim=0)

                padded_pixel_values_tiles.append(x)

            padded_pixel_values_tiles = torch.cat(padded_pixel_values_tiles, dim=0)
            data_dict['pixel_values_tiles'] = padded_pixel_values_tiles

        if len(convnext_pixel_values_tiles) > 0:
            if len(convnext_pixel_values_thumbnail) > 0:
                max_tiles = max(num_pixel_values) - 1
            else:
                max_tiles = max(num_pixel_values)

            padded_convnext_pixel_values_tiles = []

            for x in convnext_pixel_values_tiles:
                if x.shape[0] != max_tiles:
                    num_padding = max_tiles - x.shape[0]
                    x = torch.cat([x, torch.zeros(num_padding, 3, x.shape[2], x.shape[3])], dim=0)

                padded_convnext_pixel_values_tiles.append(x)

            padded_convnext_pixel_values_tiles = torch.cat(padded_convnext_pixel_values_tiles, dim=0)
            data_dict['convnext_pixel_values_tiles'] = padded_convnext_pixel_values_tiles

            
        if 'pixel_values' not in data_dict and 'pixel_values_tiles' in data_dict:
            data_dict['pixel_values'] =  data_dict['pixel_values_tiles']
            del data_dict['pixel_values_tiles']

        if 'convnext_pixel_values' not in data_dict and 'convnext_pixel_values_tiles' in data_dict:
            data_dict['convnext_pixel_values'] = data_dict['convnext_pixel_values_tiles']
            del data_dict['convnext_pixel_values_tiles']

        data_dict['num_pixel_values'] = num_pixel_values
        data_dict['max_tiles'] = max(num_pixel_values)

        
        # print(f"{'=' * 50} rank {dist.get_rank()} dynamicimages_collate_fn {'=' * 50}")
        # print(f"data_dict['pixel_values']: {data_dict['pixel_values'].shape}")
        # print(f"data_dict['pixel_values_tiles']: {data_dict['pixel_values_tiles'].shape}")
        # # print(f"data_dict['convnext_pixel_values']: {data_dict['convnext_pixel_values'].shape}")
        # # print(f"data_dict['convnext_pixel_values_tiles']: {data_dict['convnext_pixel_values_tiles'].shape}")
        # print(f"data_dict['image_size']: {data_dict['image_size']}")
        # print(f"data_dict['num_pixel_values']: {data_dict['num_pixel_values']}")
        # print(f"data_dict['max_tiles']: {data_dict['max_tiles']}")
        # # quit()

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
