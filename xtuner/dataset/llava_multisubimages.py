# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import gc
from pathlib import Path
import cv2
import numpy as np
import math

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
from xtuner.dataset.utils import expand2square, expand_original_aspect
from mmengine.dist import sync_random_seed
from torch import distributed as dist

from transformers.image_transforms import normalize, rescale


def image_transform(image, input_size, image_mean, image_std):
    image = cv2.resize(image, (input_size, input_size)).transpose((2, 0, 1))
    # print(f"resize: {image.shape, image.mean(), image.max()}")
    image = rescale(image, scale=1/255)
    # print(f"rescale: {image.shape, image.mean(), image.max()}")
    image = normalize(image, mean=image_mean, std=image_std)
    # print(f"normalize: {image.shape, image.mean(), image.max()}")
    image = torch.tensor(image)
    return image

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):

    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(
    image, 
    min_num=1, 
    max_num=6, 
    image_size=448, 
    use_thumbnail=False, 
    thumbnail_first=False,
    use_rounded_down_ratio=False
):

    orig_height, orig_width, _ = image.shape
    aspect_ratio = orig_width / orig_height

    # print(f"orig_width, orig_height: {orig_width, orig_height}, aspect_ratio: {aspect_ratio}")

    if use_rounded_down_ratio:
        # max_tiles = math.ceil(min(orig_height, orig_width) / image_size) * math.ceil(max(orig_height, orig_width) / image_size) 
        first_min_num, first_max_num = 4, 12

        # print(f"first_min_num, first_max_num: {first_min_num, first_max_num}")

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(first_min_num, first_max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= first_max_num and i * j >= first_min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # print(f"target_ratios: {target_ratios}")

        # find the closest aspect ratio to the target
        prior_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        
        # print(f"prior_aspect_ratio: {prior_aspect_ratio}")

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        new_target_ratios = []
        for i in target_ratios:
            if prior_aspect_ratio[0]%i[0] or prior_aspect_ratio[1]%i[1]:
                new_target_ratios.append(i)
            else:
                continue

        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, new_target_ratios, orig_width, orig_height, image_size)

        # print(f"rounded_down_ratio: {target_aspect_ratio}")

    else:
        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # print(f"target_ratios: {target_ratios}")

        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        
        # print(f"target_aspect_ratio: {target_aspect_ratio}")

    # print(f"target_aspect_ratio: {target_aspect_ratio}")

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # print(f"target_width, target_height: {target_width, target_height}")
    # print(f"blocks: {blocks}")

    # resize the image
    resized_img = cv2.resize(image, (target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # print(f"box: {box}")
        x1, y1, x2, y2 = box
        # split the image
        split_img = resized_img[y1:y2, x1:x2]
        # cv2.imwrite(f"/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/split_imgs_2/cv2_split_img{i}.jpg", split_img)
        processed_images.append(split_img)
    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = cv2.resize(image, (image_size, image_size))
        # cv2.imwrite(f"/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/split_imgs_2/thumbnail_img.jpg", thumbnail_img)

        if thumbnail_first:
            processed_images = [thumbnail_img] + processed_images
        else:
            processed_images.append(thumbnail_img)

    return processed_images

def slice_image_pil(image, window_size, stride):
    width, height, _ = image.shape
    # print(f"image: {image.shape}")
    split_images = []
    for y in range(0, height - window_size[1] + 1, stride):
        for x in range(0, width - window_size[0] + 1, stride):
            box = (x, y, x + window_size[0], y + window_size[1])
            # print(f"box: {box}")
            x1, y1, x2, y2 = box
            split_img = image[x1:x2, y1:y2]
            # print(split_img.shape)
            split_images.append(split_img)
            # cv2.imwrite(f'split_img_{str(box)}.jpg', split_img)
    return split_images

def slidingwindow_preprocess(image, image_size, background_color=0):
    # print(f"ori image: {image.shape}")
    width, height, _ = image.shape
    
    # if  width * height < image_size ** 2:
    #     return None

    # if min(width, height) > 448:
    #     image_size = math.ceil(min(width, height) /2)
    #     # print(f"image_size: {image_size}")

    images = []
    image, _ = expand_original_aspect(image, image_size, background_color)
    # cv2.imwrite(f'thumbnail.jpg', image)
    # images.append(image)
    # print(f"image: {image.shape}")

    # if  width * height < image_size ** 2:
    #     image_size = image_size // 2

    window_size = (image_size, image_size)
    # print(f"window_size: {window_size}")
    stride = image_size
    # print(f"window_size: {window_size}")
    windows = slice_image_pil(image, window_size, stride)
    images.extend(windows)

    return images

class LLaVAMultiSubImagesDataset(Dataset):

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
                 max_num=6,
                 thumbnail_first=False,
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

                print_log(f"llm_model_name: {llm_model_name}", logger='current')

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

        self.max_num = max_num
        self.thumbnail_first = thumbnail_first
    
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

            # print(f"image: {image.shape}")

            try:
                input_size = image_processor.crop_size['height']
            except Exception as e:
                input_size = image_processor.crop_size
                
            images = dynamic_preprocess(image, image_size=input_size, max_num=self.max_num, use_thumbnail=True, thumbnail_first=self.thumbnail_first) 
            
            pixel_values = image_processor.preprocess(images, return_tensors='pt')['pixel_values']

            num_pixel_values = len(images)

            num_padding = self.max_num + 1 - len(pixel_values)
            if num_padding > 0:
                pixel_values = torch.cat([pixel_values, torch.zeros(num_padding, 3, input_size, input_size)], dim=0)

        except Exception as e:
            print(f"{'-'*50}\nerror {e}\nin reding bad image: {os.path.join(self.image_folder, image_file)}\n{'-'*50}")
            try:
                input_size = image_processor.crop_size['height']
            except Exception as e:
                input_size = image_processor.crop_size
            pixel_values = torch.zeros(self.max_num + 1, 3, input_size, input_size)
            num_pixel_values = self.max_num + 1

        return pixel_values, num_pixel_values

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
                data_dict['pixel_values'], data_dict['num_pixel_values'] = self.encode_image(image_file, self.image_processor)

                if self.dino_image_processor:
                    data_dict['dino_pixel_values'] = self.encode_image(image_file, self.dino_image_processor)[0]

                if self.convnext_image_processor:
                    data_dict['convnext_pixel_values'] = self.encode_image(image_file,  self.convnext_image_processor)[0]


            if isinstance(image_file, list):
                images_tuple = [self.encode_image(img, self.image_processor) for img in image_file]
                
                raise NotImplementedError("todo")

        else:
            try:
                input_size = self.image_processor.crop_size['height']
            except Exception as e:
                input_size = self.image_processor.crop_size

            data_dict['pixel_values'], data_dict['num_pixel_values'] = torch.zeros(self.max_num + 1, 3, input_size, input_size), self.max_num + 1
            
            if self.dino_image_processor:
                data_dict['dino_pixel_values'] = torch.zeros(self.max_num + 1, 3, input_size, input_size)

            if self.convnext_image_processor:
                try:
                    input_size = self.convnext_image_processor.crop_size['height']
                except Exception as e:
                    input_size = self.convnext_image_processor.crop_size
                data_dict['convnext_pixel_values'] = torch.zeros(self.max_num + 1, 3, input_size, input_size)

        return data_dict

if __name__ == "__main__":

    # from transformers import (AutoModelForCausalLM, AutoTokenizer,
    #                           BitsAndBytesConfig, 
    #                           CLIPImageProcessor, 
    #                           CLIPVisionModel, 
    #                           SiglipImageProcessor, SiglipVisionModel, 
    #                           AutoModel, AutoConfig, AutoProcessor)
    # from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
    # from xtuner.utils import PROMPT_TEMPLATE
    # max_length = 4096

    # prompt_template = PROMPT_TEMPLATE.vicuna

    # llm_name_or_path='/mnt/pfs-guan-ssai/cv/sunhaoyi/dolphin-2.9.1-yi-1.5-34b'
    # visual_encoder_name_or_path = '/mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384'

    # data_root = '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/data/llava_data/'
    # data_path = data_root + 'LLaVA-Pretrain/toy.json'
    # image_folder = ''

    # tokenizer = dict(
    #     type=AutoTokenizer.from_pretrained,
    #     pretrained_model_name_or_path=llm_name_or_path,
    #     trust_remote_code=True,
    #     padding_side='right')

    # image_processor = dict(
    #     type=CLIPImageProcessor.from_pretrained,
    #     pretrained_model_name_or_path=visual_encoder_name_or_path,
    #     trust_remote_code=True,
    #     crop_size=448, 
    #     size=448
    # )

    # llava_dataset = LLaVAMultiSubImagesDataset(
    #     data_path=data_path,
    #     image_folder=image_folder,
    #     tokenizer=tokenizer,
    #     image_processor=image_processor,
    #     dataset_map_fn=llava_map_fn,
    #     template_map_fn=dict(
    #         type=template_map_fn_factory, template=prompt_template),
    #     max_length=max_length,
    #     pad_image_to_square=True,
    #     work_dir='dolphin'
    # )


    # # for idx, i in enumerate(llava_dataset):
    # #     print(idx, i['num_pixel_values'])
    # #     if i['num_pixel_values'] == 1: break

    # item = llava_dataset[0]
    # print(item['num_pixel_values'], item['pixel_values'].shape)
    image = cv2.imread('/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/view.jpg').astype(np.uint8)
    # image = cv2.resize(image, (1920, 1081))
    # print(f"image: {image.shape}")
    images = slidingwindow_preprocess(image, image_size=640) 
    # images = dynamic_preprocess(image, image_size=448, max_num=7, min_num=3, use_rounded_down_ratio=True) 
    print(f"images: {len(images)}")
