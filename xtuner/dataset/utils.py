# Copyright (c) OpenMMLab. All rights reserved.
import base64
import copy
import io
from io import BytesIO
from itertools import chain
import math

import requests
from PIL import Image

from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX

def find_last_index_of_sequence(lst, sequence):
    # 将要查找的序列转换为元组
    sequence_tuple = tuple(sequence)
    # 从列表末尾开始遍历
    for i in range(len(lst) - 2, -1, -1):
        # 如果找到与序列相匹配的子序列
        if tuple(lst[i:i+3]) == sequence_tuple:
            # 返回最后一个元素的下标
            return i + 2
    # 如果未找到匹配的子序列，则返回 -1
    return -1


def encode_fn(example,
              tokenizer,
              max_length,
              input_ids_with_output=True,
              with_image_token=False, 
              check_eos=False
):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """

    if tokenizer.__class__.__name__ == 'QWenTokenizer' or tokenizer.__class__.__name__ == 'Qwen2TokenizerFast':
        bos_token_id = []
        eos_token_id = tokenizer.eos_token_id
        assert eos_token_id is not None, \
            'Please set eos_token for Qwen tokenizer!'
    elif tokenizer.__class__.__name__ == 'ChatGLMTokenizer':
        bos_token_id = [64790, 64792]
        eos_token_id = tokenizer.eos_token_id
    else:
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
    if isinstance(bos_token_id, int):
        bos_token_id = [bos_token_id]
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    is_multi_turn_conversation = len(example['conversation']) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    next_needs_bos_token = True
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        if DEFAULT_IMAGE_TOKEN in input and with_image_token:
            chunk_encode = [
                tokenizer.encode(chunk, add_special_tokens=False)
                for chunk in input.split('<image>')
            ]
            # assert len(chunk_encode) == 2
            input_encode = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                input_encode.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    input_encode.append(IMAGE_TOKEN_INDEX)
        else:
            input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        if input_ids_with_output:
            # Add output (with loss)
            output = single_turn_conversation['output']
            output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation['need_eos_token']:
                next_needs_bos_token = True
                input_ids += eos_token_id
                labels += copy.deepcopy(eos_token_id)
            else:
                next_needs_bos_token = False
            # Add SEP (without loss)
            sep = single_turn_conversation['sep']
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    #     if 'MindGPT' in tokenizer.name_or_path:
    #         index = find_last_index_of_sequence(input_ids, [2, 29871, 13])
    #         if index == -1:
    #             input_ids = input_ids + [2, 29871, 13]
    #             labels = labels + [2, -100, -100]
    #         else:
    #             input_ids = input_ids[:index+1]
    #             labels = labels[:index+1]

    # if 'MindGPT' in tokenizer.name_or_path:
    #     assert input_ids[-3:] == [2, 29871, 13]

    return {'input_ids': input_ids, 'labels': labels}


class Packer:
    # modified from
    # https://github.com/facebookresearch/llama-recipes/blob/main/ft_datasets/utils.py

    def __init__(self, chunk_size=2048):
        self.chunk_size = chunk_size
        self.residual = {'input_ids': [], 'labels': []}

    def __call__(self, batch):

        concatenated_samples = {
            k: v + list(chain(*batch[k]))
            for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(
            concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i:i + self.chunk_size]
                    for i in range(0, chunk_num *
                                   self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size):]
                for k, v in concatenated_samples.items()
            }
        else:
            result = {k: [v] for k, v in concatenated_samples.items()}
            self.residual = {k: [] for k in concatenated_samples.keys()}

        return result


class InternRepoPacker:
    """Only used for packing data in InternLM repo
    (https://github.com/InternLM/InternLM) format."""

    def __init__(self, chunk_size=2048):
        self.chunk_size = chunk_size
        self.residual = []

    def __call__(self, batch):
        concatenated_samples = self.residual + list(chain(*batch['input_ids']))

        total_length = len(concatenated_samples)

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            input_ids = [
                concatenated_samples[i:i + self.chunk_size]
                for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
            ]
            result = {'input_ids': input_ids}
            self.residual = concatenated_samples[(chunk_num *
                                                  self.chunk_size):]
        else:
            input_ids = [concatenated_samples]
            result = {'input_ids': input_ids}
            self.residual = []

        return result


# def expand2square(pil_img, background_color):
#     width, height = pil_img.size
#     if width == height:
#         return pil_img
#     elif width > height:
#         result = Image.new(pil_img.mode, (width, width), background_color)
#         result.paste(pil_img, (0, (width - height) // 2))
#         return result
#     else:
#         result = Image.new(pil_img.mode, (height, height), background_color)
#         result.paste(pil_img, ((height - width) // 2, 0))
#         return result

import numpy as np 
import cv2 
def expand2square(cv2_img, background_color):
    background_color = np.array(background_color).astype(cv2_img.dtype)
    height, width, ch = cv2_img.shape
    if width == height:
        return cv2_img
    elif width > height:
        result = np.zeros((width, width, ch)).astype(cv2_img.dtype)
        result[:, :] = background_color
        result[(width-height)//2: (width-height)//2+height, :] = cv2_img
        return result
    else:
        result = np.zeros((height, height, ch)).astype(cv2_img.dtype)
        result[:, :] = background_color
        result[:, (height-width)//2: (height-width)//2+width] = cv2_img
        return result

def expand_original_aspect(cv2_img, image_size, background_color=0):
    background_color = np.array(background_color).astype(cv2_img.dtype)
    height, width, ch = cv2_img.shape

    if height > 1792 or width > 1792:
        if height > width:
            scale = 1792 / height
        else:
            scale = 1792 / width

        height = int(height * scale)
        width = int(width * scale)

        cv2_img = cv2.resize(cv2_img, (width, height))

    padding_height = int(math.ceil(height / image_size) * image_size)
    padding_width = int(math.ceil(width / image_size) * image_size)

    if width == padding_width and height == padding_height:
        return cv2_img, (padding_height, padding_width)

    result = np.zeros((padding_height, padding_width, ch)).astype(cv2_img.dtype)
    result[:, :] = background_color
    
    result[(padding_height-height)//2: (padding_height-height)//2+height, (padding_width-width)//2: (padding_width-width)//2+width, :] = cv2_img

    return result, (padding_height, padding_width)

def load_image(image_file):

    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        # image = Image.open(BytesIO(response.content)).convert('RGB')
        image = cv2.imread(BytesIO(response.content)).astype(np.uint8)
    else:
        # image = Image.open(image_file).convert('RGB')
        image = cv2.imread(image_file).astype(np.uint8)
    return image


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    # image = Image.open(io.BytesIO(image_data))
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


# if __name__ == "__main__":
#     cv2_img = cv2.imread('/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/demo.jpg').astype(np.uint8)
#     cv2_img = expand_original_aspect(cv2_img, 448)
#     print(cv2_img.shape)
#     cv2.imwrite('/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/processed.jpg', cv2_img)



