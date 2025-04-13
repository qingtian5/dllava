# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import gc
from pathlib import Path
import cv2
import math
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
from xtuner.dataset.huggingface import process_hf_dataset, process, process_list_dataset #, process_sharding_files
from xtuner.dataset.huggingface_2 import process_sharding_files
from xtuner.dataset.utils import expand2square, expand_original_aspect
from mmengine.dist import sync_random_seed
from torch import distributed as dist

from transformers.image_transforms import normalize, rescale

try:
    from .llava_multisubimages import dynamic_preprocess, slidingwindow_preprocess
except:
    from xtuner.dataset.llava_multisubimages import dynamic_preprocess, slidingwindow_preprocess

Image.MAX_IMAGE_PIXELS = 300000000

class LLaVADynamicImagesDataset(Dataset):

    def __init__(self,
                 data_path,
                 image_folder,
                 tokenizer,
                 dynamic_image_processor=None,
                 square_image_processor=None,
                #  dino_image_processor=None,
                 convnext_image_processor=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 system=None,
                 template_map_fn=None,
                 max_length=2048,
                 max_num=6,
                 min_num=1,
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
                    if 'mindgpt-moe-2.0-32k-vicuna' in work_dir.split('/')[-1]:
                        llm_model_name = 'mindgpt_moe_vicuna'  
                    else:
                        llm_model_name = 'mindgpt_moe_chat'  
                elif 'mindgptv6' in work_dir:
                    llm_model_name = 'mindgptv6' 
                elif 'nh_yi_34b' in work_dir:
                    llm_model_name = 'nh_yi_34b'
                elif 'dolphin' in work_dir:
                    llm_model_name = 'dolphin_yi_34b'
                elif 'qwen2_72b' in work_dir:
                    llm_model_name = 'qwen2_72b'
                elif 'qwen2_57b_a14b_instruct' in work_dir:
                    llm_model_name = 'qwen2_57b_a14b_instruct'
                elif 'qwen2_5_32b_instruct' in work_dir:
                    llm_model_name = 'qwen2_5_32b_instruct'
                elif 'qwen2_5_7b_instruct' in work_dir:
                    llm_model_name = 'qwen2_5_7b_instruct'
                else:
                    raise NotImplementedError("unkown llm")

                print_log(f"llm_model_name: {llm_model_name}", logger='current')

                processed_data_dir = f"/mnt/pfs-mc0p4k/cv/team/lishanshan/data/Pretrain_Try/processed_datasets/{llm_model_name}"
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
                        with_image_token=True, 
                        system=system)

                    json.dump(self.text_data.to_list(), open(processed_data_path, 'w'))
                    print_log(f"dump processed dataset to {processed_data_path}", logger='current')
                
                print_log("process_hf_dataset successfully return text data", logger='current')


        self.image_folder = image_folder
        if isinstance(dynamic_image_processor, dict) or isinstance(dynamic_image_processor, Config) or isinstance(dynamic_image_processor, ConfigDict):
            self.dynamic_image_processor = BUILDER.build(dynamic_image_processor)
        else:
            self.dynamic_image_processor = dynamic_image_processor

        if isinstance(square_image_processor, dict) or isinstance(square_image_processor, Config) or isinstance(square_image_processor, ConfigDict):
            self.square_image_processor = BUILDER.build(square_image_processor)
        else:
            self.square_image_processor = square_image_processor

        print_log(f"self.dynamic_image_processor:\n{self.dynamic_image_processor}", logger='current')
        print_log(f"self.square_image_processor:\n{self.square_image_processor}", logger='current')

        # if isinstance(dino_image_processor, dict) or isinstance(dino_image_processor, Config) or isinstance(dino_image_processor, ConfigDict):
        #     self.dino_image_processor = BUILDER.build(dino_image_processor)
        # else:
        #     self.dino_image_processor = dino_image_processor

        if isinstance(convnext_image_processor, dict) or isinstance(convnext_image_processor, Config) or isinstance(convnext_image_processor, ConfigDict):
            self.convnext_image_processor = BUILDER.build(convnext_image_processor)
            print_log(f"self.convnext_image_processor:\n{self.convnext_image_processor}", logger='current')
        else:
            self.convnext_image_processor = convnext_image_processor

        self.pad_image_to_square = pad_image_to_square

        self.max_num = max_num
        self.min_num = min_num
    
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
    
    def read_image(self, image_file):
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

        except Exception as e:
            print(f"{'-'*50}\nerror {e}\nin reding bad image: {os.path.join(self.image_folder, image_file)}\n{'-'*50}")

            image = None
        
        return image
    
    def encode_image_thumbnail(self, image_file, image_processor, input_size=None):
        image = self.read_image(image_file)

        if input_size is None:
            try:
                input_size = image_processor.crop_size['height']
            except Exception as e:
                input_size = image_processor.crop_size

        if image is not None:

            # print(f"input_size: {input_size}")
            
            # if image_processor.__class__.__name__ == "Dynamic_CV_CLIPImageProcessor":
            #     pass
            # else:
            #     image, _ = expand_original_aspect(image, image_size=input_size, background_color=tuple(int(x * 255) for x in image_processor.image_mean))
                
            pixel_values_thumbnail = image_processor.preprocess(image, return_tensors='pt')['pixel_values']

            num_pixel_values = 1

            # print(f"image: {image.shape}, pixel_values_thumbnail: {pixel_values_thumbnail.shape}")

        else:
            pixel_values_thumbnail = torch.zeros(1, 3, input_size, input_size)
            num_pixel_values = 1

        return pixel_values_thumbnail, num_pixel_values
    
    def encode_image_tiles(self, image_file, image_processor, input_size=None):
        image = self.read_image(image_file)

        if input_size is None:
            try:
                input_size = image_processor.crop_size['height']
            except Exception as e:
                input_size = image_processor.crop_size

        # print(f"input_size: {input_size}")

        if image is not None:
            image_tiles1 = slidingwindow_preprocess(image, image_size=input_size, background_color=tuple(int(x * 255) for x in image_processor.image_mean))
            image_tiles2 = dynamic_preprocess(image, min_num=self.min_num, max_num=self.max_num, image_size=input_size, use_rounded_down_ratio=True) 

            # if image_tiles1 is None:
            #     # print(f"input_size: {input_size}, image: {image.shape}, image_tiles1: None, image_tiles2: {len(image_tiles2)}")
            #     pixel_values_tiles2 = image_processor.preprocess(image_tiles2, return_tensors='pt')['pixel_values']
            #     pixel_values_tiles = pixel_values_tiles2
            #     num_pixel_values = len(image_tiles2)
            # else:
            #     # print(f"input_size: {input_size}, image: {image.shape}, image_tiles1: {len(image_tiles1)}, image_tiles2: {len(image_tiles2)}")
            #     pixel_values_tiles1 = image_processor.preprocess(image_tiles1, return_tensors='pt')['pixel_values']
            #     pixel_values_tiles2 = image_processor.preprocess(image_tiles2, return_tensors='pt')['pixel_values']
            #     pixel_values_tiles = torch.cat([pixel_values_tiles1, pixel_values_tiles2], dim=0)
            #     num_pixel_values = len(image_tiles1) + len(image_tiles2)


            pixel_values_tiles1 = image_processor.preprocess(image_tiles1, return_tensors='pt')['pixel_values']
            pixel_values_tiles2 = image_processor.preprocess(image_tiles2, return_tensors='pt')['pixel_values']
            pixel_values_tiles = torch.cat([pixel_values_tiles1, pixel_values_tiles2], dim=0)
            num_pixel_values = len(image_tiles1) + len(image_tiles2)

        else:
            # pixel_values_tiles = torch.zeros(self.max_num, 3, input_size, input_size)
            # num_pixel_values = self.max_num

            pixel_values_tiles = torch.zeros(1, 3, input_size, input_size)
            num_pixel_values = 1

        return pixel_values_tiles, num_pixel_values


    def __getitem__(self, index):

        if self.sharding and index != 0:
            # print(f'flag-2 sharding rank=[{torch.distributed.get_rank()}] call the __getitem__ api, index is {index}, real index is {(index - self.rank) // self.world_size}')
            index = (index - self.rank) // self.world_size
        # else:
        #     print(f'flag-2  rank=[{torch.distributed.get_rank()}] call the __getitem__ api, index is {index}')

        data_dict = self.text_data[index]

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']

            interleaved_images_input = isinstance(image_file, list) and len(image_file) > 1

            if not interleaved_images_input:
                
                if isinstance(image_file, list):
                    image_file = image_file[0]
                
                data_dict['num_pixel_values'] = 0

                if self.convnext_image_processor:
                    data_dict['convnext_pixel_values_thumbnail'], _ = self.encode_image_thumbnail(image_file, self.convnext_image_processor)
                    data_dict['convnext_pixel_values_tiles'], _ = self.encode_image_tiles(image_file, self.convnext_image_processor)

                    try:
                        input_size = self.convnext_image_processor.crop_size['height']
                    except Exception as e:
                        input_size = self.convnext_image_processor.crop_size
                else:
                    input_size = None
                    # input_size = 640

                if self.dynamic_image_processor:
                    data_dict['pixel_values_thumbnail'], num_thumbnail = self.encode_image_thumbnail(image_file, self.dynamic_image_processor, input_size)
                    data_dict['num_pixel_values'] += num_thumbnail
                else:
                    data_dict['pixel_values_thumbnail'] = None

                if self.square_image_processor:
                    data_dict['pixel_values_tiles'], num_tiles = self.encode_image_tiles(image_file, self.square_image_processor, input_size)
                    data_dict['num_pixel_values'] += num_tiles
                else:
                    data_dict['pixel_values_tiles'] = None
                
                # print(f"data_dict['pixel_values_thumbnail']: {data_dict['pixel_values_thumbnail'].shape}")
                # print(f"data_dict['pixel_values_tiles']: {data_dict['pixel_values_tiles'].shape}")
                # print(f"data_dict['num_pixel_values']: {data_dict['num_pixel_values']}")

            else:
                pixel_values_thumbnail = []
                pixel_values_tiles = []
                num_pixel_values = []

                if self.convnext_image_processor:
                    pass
                else:
                    input_size = None
                    # input_size = 640

                for img in image_file:
                    cur_num_pixel_values = 0
                    if self.dynamic_image_processor:
                        cur_pixel_values_thumbnail, num_thumbnail = self.encode_image_thumbnail(img, self.dynamic_image_processor, input_size)
                        pixel_values_thumbnail.append(cur_pixel_values_thumbnail)
                        cur_num_pixel_values += num_thumbnail
                    if self.square_image_processor:
                        # cur_pixel_values_tiles, num_tiles = self.encode_image_tiles(img, self.square_image_processor, input_size)

                        try:
                            input_size = self.square_image_processor.crop_size['height']
                        except Exception as e:
                            input_size = self.square_image_processor.crop_size

                        cur_pixel_values_tiles = torch.zeros(1, 3, input_size, input_size)
                        num_tiles = 0
                        pixel_values_tiles.append(cur_pixel_values_tiles)
                        cur_num_pixel_values += num_tiles
                    
                    num_pixel_values.append(cur_num_pixel_values)

                if len(pixel_values_thumbnail) > 0:
                    data_dict['pixel_values_thumbnail'] = pixel_values_thumbnail
                if len(pixel_values_tiles) > 0:
                    data_dict['pixel_values_tiles'] = pixel_values_tiles
                data_dict['num_pixel_values'] = num_pixel_values

                # print(f"data_dict['pixel_values_thumbnail']: {len(data_dict['pixel_values_thumbnail'])}, {[x.shape for x in data_dict['pixel_values_thumbnail']]}")
                # print(f"data_dict['pixel_values_tiles']: {len(data_dict['pixel_values_tiles'])}, {[x.shape for x in data_dict['pixel_values_tiles']]}")
                # print(f"data_dict['num_pixel_values']: {data_dict['num_pixel_values']}")
        else:
            data_dict['num_pixel_values'] = 0

            if self.convnext_image_processor:
                try:
                    input_size = self.convnext_image_processor.crop_size['height']
                except Exception as e:
                    input_size = self.convnext_image_processor.crop_size

                data_dict['convnext_pixel_values_thumbnail'] = torch.zeros(1, 3, input_size, input_size)
                data_dict['convnext_pixel_values_tiles'] = torch.zeros(1, 3, input_size, input_size)

            if self.dynamic_image_processor:
                try:
                    input_size = self.dynamic_image_processor.crop_size['height']
                except Exception as e:
                    input_size = self.dynamic_image_processor.crop_size

                data_dict['pixel_values_thumbnail'] = torch.zeros(1, 3, input_size, input_size)
                data_dict['num_pixel_values'] += 1
            else:
                data_dict['pixel_values_thumbnail'] = None

            if self.square_image_processor:
                try:
                    input_size = self.square_image_processor.crop_size['height']
                except Exception as e:
                    input_size = self.square_image_processor.crop_size

                # data_dict['pixel_values_tiles'] = torch.zeros(self.max_num, 3, input_size, input_size)
                # data_dict['num_pixel_values'] += self.max_num

                data_dict['pixel_values_tiles'] = torch.zeros(1, 3, input_size, input_size)
                data_dict['num_pixel_values'] += 1

            else:
                data_dict['pixel_values_tiles'] = None
            
            # if self.dino_image_processor:
            #     data_dict['dino_pixel_values'] = torch.zeros(self.max_num + 1, 3, input_size, input_size)

            # if self.convnext_image_processor:
            #     try:
            #         input_size = self.convnext_image_processor.crop_size['height']
            #     except Exception as e:
            #         input_size = self.convnext_image_processor.crop_size
            #     data_dict['convnext_pixel_values'] = torch.zeros(self.max_num + 1, 3, input_size, input_size)

        return data_dict

if __name__ == "__main__":
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, 
                            #   CLIPImageProcessor, 
                              CLIPVisionModel, 
                              SiglipImageProcessor, SiglipVisionModel, 
                              AutoModel, AutoConfig, AutoProcessor)
    from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
    from xtuner.utils import PROMPT_TEMPLATE
    from tqdm import tqdm

    from xtuner.utils import CV_CLIPImageProcessor as CLIPImageProcessor
    from xtuner.utils import Dynamic_CV_CLIPImageProcessor as CLIPImageProcessor

    max_length = 4096

    prompt_template = PROMPT_TEMPLATE.vicuna

    llm_name_or_path='/mnt/pfs-guan-ssai/cv/sunhaoyi/dolphin-2.9.1-yi-1.5-34b'
    visual_encoder_name_or_path = '/mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384'
    # visual_encoder_name_or_path = '/mnt/pfs-mc0p4k/cv/team/yanghongfu/hf_hub/RADIO-L'

    data_root = '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/data/llava_data/'
    data_path = data_root + 'LLaVA-Pretrain/blip_laion_cc_sbu_558k.json'
    image_folder = data_root + 'llava_images/LCS'

    # data_path = '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/data/llava_data/LLaVA-Pretrain/toy.json'
    # image_folder = ''

    tokenizer = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        padding_side='right')

    dynamic_image_processor = dict(
        type=CLIPImageProcessor.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path,
        trust_remote_code=True,
        do_resize=True,
        crop_size=448, 
        size=448
    )

    square_image_processor = dict(
        type=CLIPImageProcessor.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path,
        trust_remote_code=True,
        crop_size=448, 
        size=448
    )

    max_tiles = 7
    min_tiles = 3

    work_dir = '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/work_dirs/llava_dolphin_yi_34b_siglipdino_448_e1_gpu8_pretrain_multisubimages/dolphinyi34b-siglipdino-448-x4-x16-minimonkey'

    llava_dataset = LLaVADynamicImagesDataset(
        data_path=data_path,
        image_folder=image_folder,
        tokenizer=tokenizer,
        dynamic_image_processor=dynamic_image_processor,
        square_image_processor=square_image_processor,
        dataset_map_fn=llava_map_fn,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
        max_num=max_tiles,
        min_num=min_tiles,
        # # sampler_name='LengthGroupedSampler',
        # sampler_name='DefaultSampler',
        # group_batch_size=group_batch_size,
        # skip_filer_and_map=False,
        work_dir=work_dir
    )

    # import concurrent.futures

    # def process(idlist):
    #     num_pixel_values = []
    #     for id in tqdm(idlist):
    #         try:
    #             num_pixel_values.append((id, llava_dataset[id]['num_pixel_values']))
    #         except:
    #             print(f"\n\n\n{(id, -1)}")
    #             quit()
    #             num_pixel_values.append((id, -1))
    #     return num_pixel_values

    # num_pixel_values = []

    # idlists = [_ for _ in range(len(llava_dataset))]
    # idlists = [idlists[i::16] for i in range(16)]

    # with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    #     # 提交任务并获取 Future 对象
    #     futures = [pool.submit(process, records) for records in idlists]

    #     # 迭代 Future 对象，获取结果
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             result = future.result()  # 获取函数执行结果，如果执行出错，会抛出异常
    #         except Exception as e:
    #             print(f"{future} generated an exception: {e}", flush=True)
    #         else:
    #             # print(result, flush=True)  # 处理函数执行结果
    #             num_pixel_values += result

    # bad = []
    # for i in num_pixel_values:
    #     if i[1] == -1:
    #         bad.append(i[0])

    # print(bad)

    for i in tqdm(llava_dataset):
        # print(idx, i['num_pixel_values'])
        # num_pixel_values.append(i['num_pixel_values'])
        print(i['num_pixel_values'])
        break

    # data = llava_dataset[1]

    # print(data['pixel_values_thumbnail'].shape)
    # print(data['pixel_values_tiles'].shape)
    # print(data['num_pixel_values'])
    

