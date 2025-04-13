# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import io
import gc
from pathlib import Path
import cv2
import numpy as np

import torch
import datasets as hf_datasets
from datasets import Dataset as HFDataset
from datasets import DatasetDict, concatenate_datasets
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, SiglipImageProcessor

from xtuner.registry import BUILDER
from .huggingface import process_hf_dataset, process, process_list_dataset, process_sharding_files
from .utils import expand2square, decode_base64_to_image
from mmengine.dist import sync_random_seed
from torch import distributed as dist

from .llava_multisubimages import dynamic_preprocess

class LLaVARLAIFVDataset(Dataset):

    def __init__(self,
                #  data_path,
                 data_dir,
                 image_folder,
                 tokenizer,
                 image_processor,
                 dino_image_processor=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False, 
                 sampler_name='DefaultSampler',
                 group_batch_size=0,
                 skip_filer_and_map=False,
                 seed=None,
                 work_dir=None):
        super().__init__()

        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        data_path = [file for file in os.listdir(data_dir) if file.endswith('.parquet') and 'logp' in file]
        self.data_path = data_dir

        if len(data_path) == 0:
            raise Exception("Complete inference_logp first")
        else:
            self.data = hf_datasets.load_dataset(data_dir)['train'].cast_column("image", hf_datasets.Image(decode=False))

        self.image_folder = image_folder
        if isinstance(image_processor, dict) or isinstance(image_processor, Config) or isinstance(image_processor, ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        if isinstance(dino_image_processor, dict) or isinstance(dino_image_processor, Config) or isinstance(dino_image_processor, ConfigDict):
            self.dino_image_processor = BUILDER.build(dino_image_processor)
        else:
            self.dino_image_processor = dino_image_processor

        print_log(f"self.image_processor:\n{self.image_processor}", logger='current')
        print_log(f"self.dino_image_processor:\n{self.dino_image_processor}", logger='current')

        self.pad_image_to_square = pad_image_to_square

    def __len__(self):
        return len(self.data)

    def encode_image(self, image, max_num, image_processor):
        try:
            input_size = image_processor.crop_size['height']
        except Exception as e:
            input_size = image_processor.crop_size
            
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num) 
        
        pixel_values = image_processor.preprocess(images, return_tensors='pt')['pixel_values']

        num_pixel_values = len(images)

        num_padding = max_num + 1 - len(pixel_values)
        if num_padding > 0:
                pixel_values = torch.cat([pixel_values, torch.zeros(num_padding, 3, input_size, input_size)], dim=0)

        return pixel_values, num_pixel_values
    
    def __getitem__(self, index):

        sample = self.data[index]

        question = {'from': 'human', 'value': f"<image>\n{sample['question']}"}
        chosen = {'from': 'gpt', 'value': sample['chosen']}
        rejected = {'from': 'gpt', 'value': sample['rejected']}
        image = decode_base64_to_image(sample['image']['bytes'])
        pixel_values, num_pixel_values = self.encode_image(image, self.max_num, self.image_processor)

        metainfo = {
            "origin_dataset": sample['origin_dataset'],
            "origin_split": sample['origin_split'],
            "origin_idx": sample['idx'],
            "image_id": sample['image_path'],
        }

        data_dict = {
            'pixel_values': pixel_values,
            'num_pixel_values': num_pixel_values,
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "idx": sample['idx'],
            "metainfo": metainfo
        }
        logps=json.loads(sample['logps'])

        if type(logps) == type([]):
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
            data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = logps
        else:
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
            data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = logps['logps']
    
        return data_dict



        if self.sharding and index != 0:
            # print(f'flag-2 sharding rank=[{torch.distributed.get_rank()}] call the __getitem__ api, index is {index}, real index is {(index - self.rank) // self.world_size}')
            index = (index - self.rank) // self.world_size
        # else:
        #     print(f'flag-2  rank=[{torch.distributed.get_rank()}] call the __getitem__ api, index is {index}')

        data_dict = self.text_data[index]

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']

            if isinstance(image_file, str):

                image = self.encode_image(image_file, self.image_processor).unsqueeze(0)
                data_dict['pixel_values'] = image

                if self.dino_image_processor:
                    data_dict['dino_pixel_values'] = self.encode_image(image_file, self.dino_image_processor).unsqueeze(0)

            if isinstance(image_file, list):

                images = [self.encode_image(img).unsqueeze(0) for img in image_file]
                data_dict['pixel_values'] = torch.cat(images, dim=0)

                if self.dino_image_processor:
                    raise NotImplementedError("todo")
                    # images_dino = [self.encode_image(img, self.dino_image_processor).unsqueeze(0) for img in image_file]
                    # data_dict['dino_pixel_values'] = torch.cat(images_dino, dim=0)
        else:
            try:
                input_size = self.image_processor.crop_size['height']
            except Exception as e:
                input_size = self.image_processor.crop_size
            
            data_dict['pixel_values'] = torch.zeros(1, 3, input_size, input_size)

            if self.dino_image_processor:
                data_dict['dino_pixel_values'] = torch.zeros(1, 3, input_size, input_size)
            
        return data_dict
