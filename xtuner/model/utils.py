# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import List, Optional

import torch
from torch import nn
from torch import distributed as dist
from mmengine import print_log
from mmengine.utils.misc import get_object_from_string
from peft import PeftType
from transformers import PreTrainedModel

from xtuner.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX


def set_obj_dtype(d):
    for key, value in d.items():
        if value in ['torch.float16', 'torch.float32', 'torch.bfloat16']:
            d[key] = getattr(torch, value.split('.')[-1])


def traverse_dict(d):
    if isinstance(d, dict):
        set_obj_dtype(d)
        for key, value in d.items():
            if isinstance(value, dict):
                traverse_dict(value)
                if 'type' in value:
                    builder = value.pop('type')
                    if isinstance(builder, str):
                        builder = get_object_from_string(builder)
                    new_value = builder(**value)
                    d[key] = new_value
                    print_log(f'{key} convert to {builder}')
    elif isinstance(d, list):
        for element in d:
            traverse_dict(element)


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    if 'output_layer' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('output_layer')
    return list(lora_module_names)

def find_all_linear_names_with_lm_head(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # if 'lm_head' in lora_module_names:  # needed for 16-bit
    #     lora_module_names.remove('lm_head')
    if 'output_layer' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('output_layer')
    return list(lora_module_names)

def find_lm_head_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if 'lm_head' not in name: continue
        if isinstance(module, nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # if 'lm_head' in lora_module_names:  # needed for 16-bit
    #     lora_module_names.remove('lm_head')
    if 'output_layer' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('output_layer')

    return list(lora_module_names)

def find_all_linear_conv_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if 'stem' in name: continue

        if isinstance(module, nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if isinstance(module, nn.Conv2d):
            lora_module_names.add(name)
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    if 'output_layer' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('output_layer')
    return list(lora_module_names)


class LoadWoInit:
    """Context manager that disable parameter initialization."""

    def __init__(self):
        self.constant_ = torch.nn.init.constant_
        self.zeros_ = torch.nn.init.zeros_
        self.ones_ = torch.nn.init.ones_
        self.uniform_ = torch.nn.init.uniform_
        self.normal_ = torch.nn.init.normal_
        self.kaiming_uniform_ = torch.nn.init.kaiming_uniform_
        self.kaiming_normal_ = torch.nn.init.kaiming_normal_

    def __enter__(self, *args, **kwargs):
        torch.nn.init.constant_ = lambda *args, **kwargs: None
        torch.nn.init.zeros_ = lambda *args, **kwargs: None
        torch.nn.init.ones_ = lambda *args, **kwargs: None
        torch.nn.init.uniform_ = lambda *args, **kwargs: None
        torch.nn.init.normal_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None

    def __exit__(self, *args, **kwargs):
        torch.nn.init.constant_ = self.constant_
        torch.nn.init.zeros_ = self.zeros_
        torch.nn.init.ones_ = self.ones_
        torch.nn.init.uniform_ = self.uniform_
        torch.nn.init.normal_ = self.normal_
        torch.nn.init.kaiming_uniform_ = self.kaiming_uniform_
        torch.nn.init.kaiming_normal_ = self.kaiming_normal_


def get_peft_model_state_dict(model, state_dict=None, adapter_name='default'):
    # Modified from `https://github.com/huggingface/peft/blob/main/src/peft/utils/save_and_load.py`  # noqa: E501

    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type == PeftType.LORA:
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`  # noqa: E501
        # to be used directly with the state dict which is necessary
        # when using DeepSpeed or FSDP
        bias = config.bias
        if bias == 'none':
            to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k}
        elif bias == 'all':
            to_return = {
                k: state_dict[k]
                for k in state_dict if 'lora_' in k or 'bias' in k
            }
        elif bias == 'lora_only':
            to_return = {}
            for k in state_dict:
                if 'lora_' in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split('lora_')[0] + 'bias'
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {
            k: v
            for k, v in to_return.items()
            if (('lora_' in k and adapter_name in k) or ('bias' in k))
        }
    else:
        # Currently we only support lora
        raise NotImplementedError
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(f'{module_name}.modules_to_save.{adapter_name}' in key
                   for module_name in model.modules_to_save):
                to_return[key] = value

    return to_return


# Modified from https://github.com/haotian-liu/LLaVA/blob/82fc5e0e5f4393a4c26851fa32c69ab37ea3b146/llava/model/llava_arch.py#L99  # noqa: E501
def prepare_inputs_labels_for_multimodal(
        llm: PreTrainedModel,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None, 
        dino_pixel_values: Optional[torch.FloatTensor] = None,
        convnext_pixel_values: Optional[torch.FloatTensor] = None,
        num_pixel_values: Optional[List[int]] = None,
        **kwargs):
    if pixel_values is None:
        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': None,
            'labels': labels
        }

    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- TODO: double check
    input_ids = [
        cur_input_ids[cur_attention_mask]
        for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]
    labels = [
        cur_labels[cur_attention_mask]
        for cur_labels, cur_attention_mask in zip(labels, attention_mask)
    ]

    new_inputs_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        # print(f"cur_input_ids: {cur_input_ids.shape}")
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        # print(f"batch_idx: {batch_idx}, num_images: {num_images}")
        if num_images == 0:
            # print(f"cur_image_idx: {cur_image_idx}")
            cur_pixel_values = pixel_values[cur_image_idx]
            # print(f"cur_pixel_values: {cur_pixel_values.shape}")
            cur_inputs_embeds_1 = llm.get_input_embeddings()(cur_input_ids)
            cur_inputs_embeds = torch.cat(
                [cur_inputs_embeds_1, cur_pixel_values[0:0]], dim=0)
            new_inputs_embeds.append(cur_inputs_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(
            cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]
            ]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_inputs_embeds = llm.get_input_embeddings()(
            torch.cat(cur_input_ids_noim))
        cur_inputs_embeds_no_im = torch.split(
            cur_inputs_embeds, split_sizes, dim=0)
        cur_new_inputs_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                # print(f"cur_image_idx: {cur_image_idx}")
                cur_pixel_values = pixel_values[cur_image_idx]
                # print(f"cur_pixel_values: {cur_pixel_values.shape}")
                cur_image_idx += 1
                cur_new_inputs_embeds.append(cur_pixel_values)
                cur_new_labels.append(
                    torch.full((cur_pixel_values.shape[0], ),
                               IGNORE_INDEX,
                               device=cur_labels.device,
                               dtype=cur_labels.dtype))

        cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_inputs_embeds.append(cur_new_inputs_embeds)
        new_labels.append(cur_new_labels)

        # print(f"new_inputs_embeds: {new_inputs_embeds[-1].shape}")

    # Combine them
    max_len = max(x.shape[0] for x in new_inputs_embeds)
    batch_size = len(new_inputs_embeds)

    new_inputs_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len),
                                   IGNORE_INDEX,
                                   dtype=new_labels[0].dtype,
                                   device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len),
                                 dtype=attention_mask.dtype,
                                 device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len),
                               dtype=position_ids.dtype,
                               device=position_ids.device)

    for i, (cur_new_embed,
            cur_new_labels) in enumerate(zip(new_inputs_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        new_inputs_embeds_padded.append(
            torch.cat((cur_new_embed,
                       torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                   dtype=cur_new_embed.dtype,
                                   device=cur_new_embed.device)),
                      dim=0))
        if cur_len > 0:
            new_labels_padded[i, :cur_len] = cur_new_labels
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(
                0,
                cur_len,
                dtype=position_ids.dtype,
                device=position_ids.device)

    new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return {
        'input_ids': None,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
        'past_key_values': past_key_values,
        'inputs_embeds': new_inputs_embeds,
        'labels': new_labels
    }


def prepare_multi_subimages_inputs_labels_for_multimodal(
        llm: PreTrainedModel,
        llm_name: str = None,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        dino_pixel_values: Optional[torch.FloatTensor] = None,
        convnext_pixel_values: Optional[torch.FloatTensor] = None,
        num_pixel_values: Optional[List[int]] = None, 
        projector_num_queries: Optional[List[int]] = None, 
        add_img_slice_special_tokens: bool = False,
        mark_slice_indices: bool = False,
        **kwargs):
    if pixel_values is None:
        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': None,
            'labels': labels
        }

    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- TODO: double check
    input_ids = [
        cur_input_ids[cur_attention_mask]
        for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]
    labels = [
        cur_labels[cur_attention_mask]
        for cur_labels, cur_attention_mask in zip(labels, attention_mask)
    ]

    new_inputs_embeds = []
    new_labels = []
    slice_indices = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        cur_slice_indices = []

        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

        # print(f"{'=' * 50} rank {dist.get_rank()} prepare_multi_subimages_inputs_labels_for_multimodal {'=' * 50}")
        # print(f"num_images: {num_images}")

        if num_images == 0:
            cur_pixel_values = pixel_values[cur_image_idx]
            cur_num_pxiel_values = num_pixel_values[batch_idx]
            if len(projector_num_queries) == 2:
                valid_visual_token_num = projector_num_queries[0] + projector_num_queries[1] * (cur_num_pxiel_values - 1)
            else:
                valid_visual_token_num = projector_num_queries[0] * cur_num_pxiel_values
            cur_pixel_values = cur_pixel_values[:valid_visual_token_num, :]
            cur_slice_indices.append([-1, -1])
            slice_indices.append(cur_slice_indices)
            cur_inputs_embeds_1 = llm.get_input_embeddings()(cur_input_ids)
            cur_inputs_embeds = torch.cat(
                [cur_inputs_embeds_1, cur_pixel_values[0:0]], dim=0)
            new_inputs_embeds.append(cur_inputs_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(
            cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]
            ]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_inputs_embeds = llm.get_input_embeddings()(
            torch.cat(cur_input_ids_noim))
        # print(f"cur_input_ids_noim: {cur_input_ids_noim}")
        cur_inputs_embeds_no_im = torch.split(
            cur_inputs_embeds, split_sizes, dim=0)
        cur_new_inputs_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_pixel_values = pixel_values[cur_image_idx]
                # print(f"1 cur_pixel_values: {cur_pixel_values.shape}")
                cur_num_pxiel_values = num_pixel_values[batch_idx]
                # print(f"cur_num_pxiel_values: {cur_num_pxiel_values}")
                if len(projector_num_queries) == 2:
                    valid_visual_token_num = projector_num_queries[0] + projector_num_queries[1] * (cur_num_pxiel_values - 1)
                    # print(f"valid_visual_token_num: {valid_visual_token_num}")

                    # print(f"add_img_slice_special_tokens: {add_img_slice_special_tokens}")
                    if add_img_slice_special_tokens:
                        # print(f"llm_name: {llm_name}")
                        
                        if 'LlamaForCausalLM' in llm_name: 
                            # dolphin-2.9.1-yi-1.5-34b
                            img_start_token = llm.get_input_embeddings()(torch.tensor([295]).to(device=llm.device))
                            img_end_token = llm.get_input_embeddings()(torch.tensor([296]).to(device=llm.device))
                            slice_start_token = llm.get_input_embeddings()(torch.tensor([297]).to(device=llm.device))
                            slice_end_token = llm.get_input_embeddings()(torch.tensor([298]).to(device=llm.device))

                            # print(f"slice_start_token: {slice_start_token.mean()}, {slice_start_token.max()}")
                            # print(f"slice_end_token: {slice_end_token.mean()}, {slice_end_token.max()}")

                        elif llm_name == 'MixtralForCausalLM': 
                            # MindGPT-2.0-32K
                            img_start_token = llm.get_input_embeddings()(torch.tensor([128212]).to(device=llm.device))
                            img_end_token = llm.get_input_embeddings()(torch.tensor([128213]).to(device=llm.device))
                            slice_start_token = llm.get_input_embeddings()(torch.tensor([128218]).to(device=llm.device))
                            slice_end_token = llm.get_input_embeddings()(torch.tensor([128219]).to(device=llm.device))

                        elif llm_name == 'Qwen2MoeForCausalLM':
                            img_start_token = llm.get_input_embeddings()(torch.tensor([151646]).to(device=llm.device))
                            img_end_token = llm.get_input_embeddings()(torch.tensor([151647]).to(device=llm.device))
                            slice_start_token = llm.get_input_embeddings()(torch.tensor([151648]).to(device=llm.device))
                            slice_end_token = llm.get_input_embeddings()(torch.tensor([151649]).to(device=llm.device))  

                        elif llm_name == 'Qwen2ForCausalLM':
                            img_start_token = llm.get_input_embeddings()(torch.tensor([151665]).to(device=llm.device))
                            img_end_token = llm.get_input_embeddings()(torch.tensor([151666]).to(device=llm.device))
                            slice_start_token = llm.get_input_embeddings()(torch.tensor([151667]).to(device=llm.device))
                            slice_end_token = llm.get_input_embeddings()(torch.tensor([151668]).to(device=llm.device))                 

                        thumbnail_pixel_values = cur_pixel_values[:projector_num_queries[0], :]
                        # print(f"thumbnail_pixel_values: {thumbnail_pixel_values.shape}")

                        if valid_visual_token_num - projector_num_queries[0] > 0: 
                            tiles_pixel_values = cur_pixel_values[projector_num_queries[0]:valid_visual_token_num, :]
                            # print(f"tiles_pixel_values: {tiles_pixel_values.shape}")
                            cur_pixel_values = torch.cat([img_start_token, thumbnail_pixel_values, img_end_token, slice_start_token, tiles_pixel_values, slice_end_token], dim=0)
                            # print(f"add img & slice tokens cur_pixel_values: {cur_pixel_values.shape}")
                        else:
                            cur_pixel_values = torch.cat([img_start_token, thumbnail_pixel_values, img_end_token], dim=0)
                            # print(f"add img tokens cur_pixel_values: {cur_pixel_values.shape}")

                    else:
                        cur_pixel_values = cur_pixel_values[:valid_visual_token_num, :]
                        # print(f"2 cur_pixel_values: {cur_pixel_values.shape}")
                else:
                    valid_visual_token_num = projector_num_queries[0] * cur_num_pxiel_values
                    # print(f"valid_visual_token_num: {valid_visual_token_num}")

                    if add_img_slice_special_tokens:
                        # print(f"llm_name: {llm_name}")
                        if llm_name == 'LlamaForCausalLM': 
                            # dolphin-2.9.1-yi-1.5-34b
                            img_start_token = llm.get_input_embeddings()(torch.tensor([295]).to(device=llm.device))
                            img_end_token = llm.get_input_embeddings()(torch.tensor([296]).to(device=llm.device))

                        elif llm_name == 'MixtralForCausalLM': 
                            # MindGPT-2.0-32K
                            img_start_token = llm.get_input_embeddings()(torch.tensor([128212]).to(device=llm.device))
                            img_end_token = llm.get_input_embeddings()(torch.tensor([128213]).to(device=llm.device))

                        elif llm_name == 'Qwen2MoeForCausalLM':
                            img_start_token = llm.get_input_embeddings()(torch.tensor([151646]).to(device=llm.device))
                            img_end_token = llm.get_input_embeddings()(torch.tensor([151647]).to(device=llm.device))

                        elif llm_name == 'Qwen2ForCausalLM':
                            img_start_token = llm.get_input_embeddings()(torch.tensor([151665]).to(device=llm.device))
                            img_end_token = llm.get_input_embeddings()(torch.tensor([151666]).to(device=llm.device))

                        cur_pixel_values = torch.cat([img_start_token, cur_pixel_values[:valid_visual_token_num, :], img_end_token], dim=0)
                        # print(f"add_img_slice_special_tokens cur_pixel_values: {cur_pixel_values.shape}")

                    else:
                        cur_pixel_values = cur_pixel_values[:valid_visual_token_num, :]
                        # print(f"2 cur_pixel_values: {cur_pixel_values.shape}")

                cur_image_idx += 1
                cur_new_inputs_embeds.append(cur_pixel_values)
                # print(f"cur_new_inputs_embeds: {len(cur_new_inputs_embeds)}, {[x.shape for x in cur_new_inputs_embeds]}")
                if len(projector_num_queries) == 2:
                    cur_slice_indices.append([
                        cur_new_inputs_embeds[-2].shape[0] + projector_num_queries[0] + 2, 
                        cur_new_inputs_embeds[-2].shape[0] + cur_pixel_values.shape[0] - 1, 
                    ])

                # else:
                #     NotImplementedError("The calculation of slice indices for the case where len(projector_num_queries) == 1 has not been implemented yet.")

                # print(f"slice_indices: {cur_slice_indices}")
                
                cur_new_labels.append(
                    torch.full((cur_pixel_values.shape[0], ),
                               IGNORE_INDEX,
                               device=cur_labels.device,
                               dtype=cur_labels.dtype))

        cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        # print(f"cur_new_inputs_embeds: {cur_new_inputs_embeds.shape}")

        # print(f"slice_indices: {cur_slice_indices}")
        # print(f"slice start: {cur_new_inputs_embeds[cur_slice_indices[0][0]].mean()}, {cur_new_inputs_embeds[cur_slice_indices[0][0]].max()}")
        # print(f"slice end: {cur_new_inputs_embeds[cur_slice_indices[0][-1]].mean()}, {cur_new_inputs_embeds[cur_slice_indices[0][-1]].max()}")

        new_inputs_embeds.append(cur_new_inputs_embeds)
        new_labels.append(cur_new_labels)
        slice_indices.append(cur_slice_indices)
    # print(f"slice_indices: {slice_indices}")
    # quit()

    # Combine them
    max_len = max(x.shape[0] for x in new_inputs_embeds)
    batch_size = len(new_inputs_embeds)

    new_inputs_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len),
                                   IGNORE_INDEX,
                                   dtype=new_labels[0].dtype,
                                   device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len),
                                 dtype=attention_mask.dtype,
                                 device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len),
                               dtype=position_ids.dtype,
                               device=position_ids.device)

    for i, (cur_new_embed,
            cur_new_labels) in enumerate(zip(new_inputs_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        new_inputs_embeds_padded.append(
            torch.cat((cur_new_embed,
                       torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                   dtype=cur_new_embed.dtype,
                                   device=cur_new_embed.device)),
                      dim=0))
        if cur_len > 0:
            new_labels_padded[i, :cur_len] = cur_new_labels
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(
                0,
                cur_len,
                dtype=position_ids.dtype,
                device=position_ids.device)

    new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    # print(f"new_inputs_embeds: {new_inputs_embeds.shape}")

    if mark_slice_indices:
        return {
            'input_ids': None,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': new_inputs_embeds,
            'labels': new_labels,
            'slice_indices': slice_indices
        }
    else:
        return {
            'input_ids': None,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': new_inputs_embeds,
            'labels': new_labels
        }



def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)


def guess_load_checkpoint(pth_model):
    if osp.isfile(pth_model):
        print_log(f"load checkpoint from torch.load", "current")
        state_dict = torch.load(pth_model, map_location='cpu')
        # state_dict = torch.load(pth_model, map_location=lambda storage, loc: storage.cuda(int(os.environ['LOCAL_RANK'])))
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    elif osp.isdir(pth_model):
        try:
            from deepspeed.utils.zero_to_fp32 import \
                get_fp32_state_dict_from_zero_checkpoint
        except ImportError:
            raise ImportError(
                'The provided PTH model appears to be a DeepSpeed checkpoint. '
                'However, DeepSpeed library is not detected in current '
                'environment. This suggests that DeepSpeed may not be '
                'installed or is incorrectly configured. Please verify your '
                'setup.')
        state_dict = get_fp32_state_dict_from_zero_checkpoint(
            osp.dirname(pth_model), osp.basename(pth_model))
    else:
        raise FileNotFoundError(f'Cannot find {pth_model}')
    return state_dict


if __name__ == "__main__":
    pretrained_pth = '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/work_dirs/llava_mindgpt_moe_siglipdino_lora_e1_gpu8_finetune_multisubimages/sft-mindgpt-v7moe-siglipdino-448-sw-minimonkey-img-slice-300m-stage2-cambrian-cauldron-great-intergration-50m-ft/iter_14400.pth/'
    pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

    for k, v in pretrained_state_dict.items():
        print(k, v.shape)

    torch.save(pretrained_state_dict, '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/work_dirs/llava_mindgpt_moe_siglipdino_lora_e1_gpu8_finetune_multisubimages/sft-mindgpt-v7moe-siglipdino-448-sw-minimonkey-img-slice-300m-stage2-cambrian-cauldron-great-intergration-50m-ft/model.bin.iter_14400')