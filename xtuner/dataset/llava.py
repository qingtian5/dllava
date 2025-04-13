# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import gc
from pathlib import Path
import cv2
import numpy as np

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, concatenate_datasets
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, SiglipImageProcessor

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, process, process_list_dataset, process_sharding_files
from xtuner.dataset.utils import expand2square
from mmengine.dist import sync_random_seed
from torch import distributed as dist


class LLaVADataset(Dataset):

    def __init__(self,
                 data_path,
                 image_folder,
                 tokenizer,
                 image_processor,
                 dino_image_processor=None,
                 convnext_image_processor=None,
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

        print_log(f"data_path: {data_path}", logger='current')
        self.sharding = False
        if isinstance(data_path, list):
            # json_data = json.load(open(data_path[0]))
            # for idx in range(len(json_data)):
            #     if isinstance(json_data[idx]['id'], int):
            #         json_data[idx]['id'] = str(json_data[idx]['id'])
            # self.text_data = HFDataset.from_list(json_data)
            # print_log(f"text_data has {len(self.text_data)} records", logger='current')
            # del json_data
            # gc.collect()
            # for dp in data_path[1:]:
            #     json_data = json.load(open(dp))
            #     for idx in range(len(json_data)):
            #         if isinstance(json_data[idx]['id'], int):
            #             json_data[idx]['id'] = str(json_data[idx]['id'])
            #     self.text_data = concatenate_datasets([self.text_data, HFDataset.from_list(json_data)])
            #     print_log(f"text_data has {len(self.text_data)} records", logger='current')
            #     del json_data
            #     gc.collect()
            self.text_data = process_list_dataset(data_path=data_path)
        else:
            # for sharding files loading
            if os.path.isdir(data_path):
                self.sampler_name = sampler_name
                if seed is None:
                    seed = sync_random_seed()
                self.sharding = True
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                self.text_data = process_sharding_files(data_path=data_path, 
                                                        sampler_name=sampler_name,
                                                        group_batch_size=group_batch_size,
                                                        tokenizer=tokenizer,
                                                        max_length=max_length,
                                                        dataset_map_fn=dataset_map_fn,
                                                        template_map_fn=template_map_fn,
                                                        skip_filer_and_map=skip_filer_and_map,
                                                        seed=seed,
                                                        work_dir=work_dir)
            else:
                if 'mindgpt_moe' in work_dir:
                    if 'mindgpt-moe-chat' in work_dir.split('/')[-1]:
                        llm_model_name = 'mindgpt_moe_chat'  
                    if 'mindgpt-moe-2.0-32k-vicuna' in work_dir.split('/')[-1]:
                        llm_model_name = 'mindgpt_moe_vicuna'  
                elif 'mindgptv6' in work_dir:
                    llm_model_name = 'mindgptv6' 
                elif 'nh_yi_34b' in work_dir:
                    llm_model_name = 'nh_yi_34b'
                elif 'dolphin' in work_dir:
                    llm_model_name = 'dolphin_yi_34b'
                elif 'qwen2_72b' in work_dir:
                    llm_model_name = 'qwen2_72b'
                else:
                    raise NotImplementedError("unkown llm")

                processed_data_dir = f"/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/data/llava_data/processed_datasets/{llm_model_name}"
                if not os.path.exists(processed_data_dir):
                    os.makedirs(processed_data_dir)

                json_name = os.path.basename(data_path)
                processed_data_path = f'{processed_data_dir}/{json_name}'

                if os.path.exists(processed_data_path):
                    json_data = json.load(open(processed_data_path))
                    for idx in range(len(json_data)):
                        if isinstance(json_data[idx]['id'], int):
                            json_data[idx]['id'] = str(json_data[idx]['id'])
                    self.text_data = HFDataset.from_list(json_data)
                    print_log(f"load processed dataset from {processed_data_path}", logger='current')
                else:
                    json_data = json.load(open(data_path))
                    for idx in range(len(json_data)):
                        if isinstance(json_data[idx]['id'], int):
                            json_data[idx]['id'] = str(json_data[idx]['id'])
                    json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
                    self.text_data = process_hf_dataset(
                        dataset=json_data,
                        tokenizer=tokenizer,
                        max_length=max_length,
                        dataset_map_fn=dataset_map_fn,
                        template_map_fn=template_map_fn,
                        split='train',
                        max_dataset_length=max_dataset_length,
                        remove_unused_columns=False,
                        pack_to_max_length=False,
                        with_image_token=True)

                    json.dump(self.text_data.to_list(), open(processed_data_path, 'w'))
                    print_log(f"dump processed dataset to {processed_data_path}", logger='current')
                
                print_log("process_hf_dataset successfully return text data", logger='current')


        self.image_folder = image_folder
        if isinstance(image_processor, dict) or isinstance(image_processor, Config) or isinstance(image_processor, ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        if isinstance(dino_image_processor, dict) or isinstance(dino_image_processor, Config) or isinstance(dino_image_processor, ConfigDict):
            self.dino_image_processor = BUILDER.build(dino_image_processor)
        else:
            self.dino_image_processor = dino_image_processor

        if isinstance(convnext_image_processor, dict) or isinstance(convnext_image_processor, Config) or isinstance(convnext_image_processor, ConfigDict):
            self.convnext_image_processor = BUILDER.build(convnext_image_processor)
        else:
            self.convnext_image_processor = convnext_image_processor


        print_log(f"self.image_processor:\n{self.image_processor}", logger='current')
        print_log(f"self.dino_image_processor:\n{self.dino_image_processor}", logger='current')
        print_log(f"self.convnext_image_processor:\n{self.convnext_image_processor}", logger='current')

        self.pad_image_to_square = pad_image_to_square

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            if 'input_ids' in data_dict:
                cur_len = len(data_dict['input_ids'])
            else:
                cur_len = 500
                if data_dict.get('image', None) is not None and len(data_dict['image']) == 1: cur_len = 100
                if data_dict.get('image', None) is not None and len(data_dict['image']) > 1: cur_len = 200
            if data_dict.get('image', None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        if self.sharding:
            return len(self.text_data) * self.world_size
        return len(self.text_data)

    def encode_image(self, image_file, image_processor):
        try:
            # image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

            image_path = os.path.join(self.image_folder, image_file)
            
            # print(f"image_path: {image_path}")

            tmp_image_path = Path(image_path)
            if not tmp_image_path.suffix.lower() == ".jpg":
                image_pil = Image.open(str(image_path)).convert('RGB')
                new_image_path = str(tmp_image_path.with_suffix(".jpg"))
                image_pil.save(new_image_path)
                    
                image = cv2.imread(new_image_path)
            else:
                image = cv2.imread(image_path)

            if image is None:
                image_pil = Image.open(image_path).convert('RGB')
                new_image_path = image_path.replace(Path(image_path).stem, f"{Path(image_path).stem}_pilsave")
                image_pil.save(new_image_path)

                image = cv2.imread(new_image_path)
            
            image = image.astype(np.uint8)

            # print(f"image: {image}")
            # quit()

            if self.pad_image_to_square and image_processor.image_mean is not None:
                image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))

            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            # print(f"image: {image}, {image.shape}")
            # quit()

        except Exception as e:

            print(f"{'-'*50}\nerror {e}\nin reading bad image: {os.path.join(self.image_folder, image_file)}\n{'-'*50}")

            try:
                input_size = image_processor.crop_size['height']
            except Exception as e:
                input_size = image_processor.crop_size

            image = torch.zeros(3, input_size, input_size)

        return image


    def __getitem__(self, index):

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

                if self.convnext_image_processor:
                    data_dict['convnext_pixel_values'] = self.encode_image(image_file, self.convnext_image_processor).unsqueeze(0)

            if isinstance(image_file, list):

                images = [self.encode_image(img, self.image_processor).unsqueeze(0) for img in image_file]
                data_dict['pixel_values'] = torch.cat(images, dim=0)
        else:
            try:
                input_size = self.image_processor.crop_size['height']
            except Exception as e:
                input_size = self.image_processor.crop_size
            
            data_dict['pixel_values'] = torch.zeros(1, 3, input_size, input_size)

            if self.dino_image_processor:
                data_dict['dino_pixel_values'] = torch.zeros(1, 3, input_size, input_size)

            if self.convnext_image_processor:
                try:
                    input_size = self.convnext_image_processor.crop_size['height']
                except Exception as e:
                    input_size = self.convnext_image_processor.crop_size

                data_dict['convnext_pixel_values'] = torch.zeros(1, 3, input_size, input_size)
            
        return data_dict
    
if __name__ == "__main__":

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, 
                              CLIPImageProcessor, 
                              CLIPVisionModel, 
                              SiglipImageProcessor, SiglipVisionModel, 
                              AutoModel, AutoConfig, AutoProcessor)
    from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
    from xtuner.utils import PROMPT_TEMPLATE

    llm_name_or_path = '/mnt/pfs-guan-ssai/cv/sunhaoyi/dolphin-2.9.1-yi-1.5-34b'
    visual_encoder_name_or_path = '/mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384'

    data_path = '/mnt/pfs-mc0p4k/cv/team/yanghongfu/sharding_data/665k_mmdu_mantis'
    image_folder = ''

    prompt_template = PROMPT_TEMPLATE.qwen_chat
    # max_length = int(2048 - (448 / 14)**2)
    # max_length = int(4096 - (448 / 14)**2)
    max_length = int(4096 - 256 * 5)
    
    batch_size = 4
    group_batch_size = 4

    tokenizer = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        padding_side='right')

    image_processor = dict(
        type=CLIPImageProcessor.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path,
        trust_remote_code=True,
        crop_size=448, 
        size=448)

    llava_dataset = LLaVADataset(
        data_path=data_path,
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        # dino_image_processor=dino_image_processor,
        dataset_map_fn=llava_map_fn,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
        # sampler_name='LengthGroupedSampler',
        sampler_name='DefaultSampler',
        group_batch_size=group_batch_size,
        skip_filer_and_map=False,
        work_dir='/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/work_dirs/llava_dolphin_yi_34b_siglipdino_lora_e1_gpu8_finetune/dolphinyi34b-siglipdino-100m-mmc4-ft-interleaved'
    )

    item = llava_dataset[0]

    print(item)