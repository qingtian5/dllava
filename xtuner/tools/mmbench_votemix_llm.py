# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
import re
import string
import time
import math
import cv2

import numpy as np
import pandas as pd
import torch
import tqdm
from huggingface_hub import snapshot_download
from mmengine import mkdir_or_exist
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from torch.utils.data import Dataset
from transformers import (AutoModel, AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, 
                          ConvNextModel,
                          CLIPImageProcessor, CLIPVisionModel, 
                          SiglipImageProcessor, SiglipVisionModel,
                          GenerationConfig)

from xtuner.model import LlamaForCausalLM_VoteMix, LlamaForCausalLM_PDrop

from xtuner.dataset.utils import decode_base64_to_image, expand2square, load_image
#, expand_original_aspect
from xtuner.dataset.llava_multisubimages import dynamic_preprocess, slice_image_pil
#, slidingwindow_preprocess
from xtuner.model.utils import prepare_inputs_labels_for_multimodal, prepare_multi_subimages_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX,
                          PROMPT_TEMPLATE)
from xtuner.tools.train import *

from xtuner.utils import CV_CLIPImageProcessor as CLIPImageProcessor

from transformers import PreTrainedModel
import timm
from einops import rearrange
from typing import List, Optional


TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def expand_original_aspect(cv2_img, image_size, background_color=0):
    background_color = np.array(background_color).astype(cv2_img.dtype)
    height, width, ch = cv2_img.shape

    if height > 2048 or width > 2048:
        if height > width:
            scale = 2048 / height
        else:
            scale = 2048 / width

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

def tuple_type(value):
    """
    Custom type conversion function to convert a string to a tuple.
    """
    parts = value.split(',')
    if len(parts) != 2:
        raise ValueError("Invalid tuple format. Expected 'string,path'.")
    return (parts[0], parts[1])

def parse_args():
    parser = argparse.ArgumentParser(description='MMBench')
    parser.add_argument(
        'model_name_or_path', help='Hugging Face model name or path')
    parser.add_argument('--data-path', default=None, help='data path')
    parser.add_argument('--work-dir', help='the dir to save results')
    parser.add_argument('--llava', default=None, help='llava name or path')
    parser.add_argument(
        '--visual-encoder', default=None, help='visual encoder name or path')
    parser.add_argument(
        '--dino-visual-encoder', default=None, help='visual encoder name or path')
    parser.add_argument(
        '--convnext-visual-encoder', default=None, help='visual encoder name or path')
    parser.add_argument(
        '--visual-select-layer', default=-2, help='visual select layer')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default='None',
        help='Specify a prompt template')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=1024,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    parser.add_argument(
        '--dynamic-image-size',
        type=int,
        default=None,
        help='Dynamic Image Size')
    parser.add_argument(
        '--convnext-image-size',
        type=int,
        default=None,
        help='Convnext Image Size')
    parser.add_argument(
        '--num-sub-images',
        type=int,
        default=0,
        help='Num of Sub Images')
    parser.add_argument(
        '--projector-num-queries',
        nargs='+', 
        type=int, 
        default=[],
        help='Num of Peojector Queries')
    parser.add_argument(
        '--fusion-channel',
        action='store_true',
        help='Enable fusion channel'
    )
    args = parser.parse_args()
    return args


class MMBenchDataset(Dataset):
    ABBRS = {
        'coarse_perception': 'CP',
        'finegrained_perception (instance-level)': 'FP-S',
        'finegrained_perception (cross-instance)': 'FP-C',
        'logic_reasoning': 'LR',
        'relation_reasoning': 'RR',
        'attribute_reasoning': 'AR',
        'sketch_reasoning': 'Sketch Reasoning',
        'scenery_building': 'Scenery & Building',
        'food_clothes': 'Food & Clothes',
        'historical_figure': 'Historical Figure',
        'traditional_show': 'Traditional Show',
        'calligraphy_painting': 'Calligraphy Painting',
        'cultural_relic': 'Cultural Relic',
        # 增加 LiAuto 类别
        'color': 'Color',
        'count': 'Count',
        'existence': 'Existence',
        'OCR': 'OCR',
        'position': 'Position',
        'scene': 'Scene' 
    }

    def __init__(self, data_file, qa_type):
        self.data_file = data_file
        self.df = pd.read_csv(data_file, sep='\t')
        self.split = 'dev' if 'answer' in self.df.iloc[0].keys() else 'test'
        self.has_l2_category = 'l2-category' in self.df.columns.to_list()
        self.qa_type = qa_type


    def get_image(self, image):
        try:
            while len(image) < 16:
                image = self.df[self.df['index'] == int(image)]['image'].values
                assert len(image) == 1
                image = image[0]
            image = decode_base64_to_image(image)
        except  Exception as e:
            return None
        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        image = self.get_image(image)

        try:
            width = self.df.iloc[idx]['width']
        except Exception as e:
            width = None

        try:
            height = self.df.iloc[idx]['height']
        except Exception as e:
            height = None

        if self.qa_type == 'LiAuto':
            question_type = self.df.iloc[idx]['type']
        else:
            question_type = self.qa_type
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[
            0].keys() else None

        try:
            category = self.df.iloc[idx]['category']
        except:
            category = None

        try:
            multi_choice_options = self.df.iloc[idx]['multi-choice options']
        except:
            multi_choice_options = None

        options = {
            cand: self.load_from_df(idx, cand)
            for cand in string.ascii_uppercase
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = ''
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        hint = self.load_from_df(idx, 'hint')

        data = {
            'img': image,
            'question': question,
            'answer': answer,
            'options': options_prompt,
            'category': category,
            'options_dict': options,
            'index': index,
            'context': hint,
            'type': question_type,
            'split': self.split,
            'width': width,
            'height': height,
            'multi-choice options': multi_choice_options
        }

        if 'conversations' in self.df.iloc[idx]:
            data['conversations'] = self.df.iloc[idx]['conversations']

        if self.has_l2_category:
            data.update({'l2-category': self.df.iloc[idx]['l2-category']})
        return data

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None

    def eval_result(self, result_df, show=True):

        def calc_acc(df, group='category'):
            assert group in ['overall', 'category', 'l2-category']
            if group == 'overall':
                res = {'Average': np.mean(df['hit'])}
            else:
                res = {}
                abilities = list(set(df[group]))
                abilities.sort()
                for ab in abilities:
                    sub_df = df[df[group] == ab]
                    ab = self.ABBRS[ab] if ab in self.ABBRS else ab
                    res[ab] = np.mean(sub_df['hit'])
            return res

        def eval_sub_data(sub_data, answer_map):
            lt = len(sub_data)
            for i in range(lt):
                item = sub_data.iloc[i]
                match = re.search(r'([A-D]+)', item['prediction'])
                pred = match.group(1) if match else ''
                gt = answer_map[item['index']]
                if gt != pred:
                    return 0
            return 1

        def show_result(ret_json):
            show_dict = ret_json.copy()
            table = Table(title=f' MMBench ({self.data_file}) ')
            console = Console()
            table.add_column('Category', justify='left')
            table.add_column('Accuracy (%)', justify='right')
            average = show_dict.pop('Average') * 100
            table.add_row('Average', f'{average:.1f}')
            table.add_section()
            for cat_name, cat_acc in show_dict.items():
                table.add_row(cat_name, f'{cat_acc * 100:.1f}')
            with console.capture() as capture:
                console.print(table, end='')
            print('\n' + capture.get())
            print('Note: Please be cautious if you use the results in papers, '
                  "since we don't use ChatGPT as a helper for choice "
                  'extraction')

        data = result_df.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        for k in data.keys():
            data[k.lower() if k not in 'ABCD' else k] = data.pop(k)

        data_main = data[data['index'] < int(1e6)]
        cate_map = {
            i: c
            for i, c in zip(self.df['index'], self.df['category'])
        }
        if self.has_l2_category:
            l2_cate_map = {
                i: c
                for i, c in zip(self.df['index'], self.df['l2-category'])
            }
        answer_map = {
            i: c
            for i, c in zip(self.df['index'], self.df['answer'])
        }

        lt = len(data_main)
        hit, tot = 0, 0
        result = {}
        for i in range(lt):
            item_main = data_main.iloc[i]
            idx = item_main['index']
            assert idx not in result
            sub_data = data[data['index'] % int(1e6) == idx]
            ret = eval_sub_data(sub_data, answer_map)
            result[idx] = ret
            hit += ret
            tot += 1

        indices = data_main['index']
        data_main = data_main.copy()
        data_main['hit'] = [result[i] for i in indices]
        main_idx = data_main['index']
        data_main['category'] = [cate_map[i] for i in main_idx]

        ret_json = calc_acc(data_main, 'overall')

        if self.has_l2_category:
            data_main['l2-category'] = [l2_cate_map[i] for i in main_idx]
            l2 = calc_acc(data_main, 'l2-category')
            ret_json.update(l2)
        else:
            leaf = calc_acc(data_main, 'category')
            ret_json.update(leaf)
        if show:
            show_result(ret_json)
        return ret_json

    def eval_li_result(self, result_df, show=True):

        def calc_acc(df, group='category'):
            assert group in ['overall', 'category', 'l2-category']
            if group == 'overall':
                res = {'Average': np.mean(df['hit'])}
            else:
                res = {}
                abilities = list(set(df[group]))
                abilities.sort()
                for ab in abilities:
                    sub_df = df[df[group] == ab]
                    ab = self.ABBRS[ab] if ab in self.ABBRS else ab
                    res[ab] = np.mean(sub_df['hit'])
            return res

        def show_result(ret_json):
            show_dict = ret_json.copy()
            table = Table(title=f'[{self.data_file}]')
            console = Console()
            table.add_column('Category', justify='left')
            table.add_column('Accuracy (%)', justify='right')
            average = show_dict.pop('Average') * 100
            table.add_row('Average', f'{average:.1f}')
            table.add_section()
            for cat_name, cat_acc in show_dict.items():
                table.add_row(cat_name, f'{cat_acc * 100:.1f}')
            with console.capture() as capture:
                console.print(table, end='')
            print('\n' + capture.get())
            print('Note: Please be cautious if you use the results in papers, '
                  "since we don't use ChatGPT as a helper for choice "
                  'extraction')

        data = result_df.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        for k in data.keys():
            data[k.lower() if k not in 'ABCD' else k] = data.pop(k)

        cate_map = {
            i: c
            for i, c in zip(self.df['index'], self.df['category'])
        }
        if self.has_l2_category:
            l2_cate_map = {
                i: c
                for i, c in zip(self.df['index'], self.df['l2-category'])
            }
        answer_map = {
            i: c
            for i, c in zip(self.df['index'], self.df['answer'])
        }

        lt = len(data)
        hit, tot = 0, 0
        result = {}
        for i in range(lt):
            item = data.iloc[i]
            idx = item['index']
            assert idx not in result
            ret = 0
            if item['type'] == "choice":
                match = re.search(r'([A-D]+)', item['prediction'])
                pred = match.group(1) if match else ''
                gt = answer_map[idx]
                if gt == pred:
                    ret = 1
            elif item['type'] == "bool":
                s = item['prediction'].lower()
                if 'yes' in s and 'no' not in s:
                    extract = 'Yes'
                elif 'yes' not in s and 'no' in s:
                    extract = 'No'
                else:
                    extract = 'Unknown'
                gt = answer_map[idx]
                if gt == extract:
                    ret = 1
            result[idx] = ret
            hit += ret
            tot += 1

        indices = data['index']
        data = data.copy()
        data['hit'] = [result[i] for i in indices]
        main_idx = data['index']
        data['category'] = [cate_map[i] for i in main_idx]

        ret_json = calc_acc(data, 'overall')

        if self.has_l2_category:
            data['l2-category'] = [l2_cate_map[i] for i in main_idx]
            l2 = calc_acc(data, 'l2-category')
            ret_json.update(l2)
        else:
            leaf = calc_acc(data, 'category')
            ret_json.update(leaf)
        if show:
            show_result(ret_json)
        return ret_json

def listinstr(lst, s):
    assert isinstance(lst, list)
    for item in lst:
        if item in s:
            return True
    return False

def DATASET_TYPE(dataset):
    # Dealing with Custom Dataset
    dataset = dataset.lower()
    if listinstr(['mmbench', 'seedbench', 'ccbench', 'mmmu', 'scienceqa', 'ai2d', 'mmstar', 'mme-realworld', 'mme-realworld-cn'], dataset):
        return 'multi-choice'
    elif listinstr(['mme', 'hallusion'], dataset):
        return 'Y/N'
    elif 'coco' in dataset:
        return 'Caption'
    elif listinstr(['ocrvqa', 'chartqa', 'mathvista', 'docvqa', 'llavabench', 'mmvet', 'ocrbench'], dataset):
        return 'VQA'
    elif 'textvqa' in dataset:
        return 'TextVQA'
    # elif '' in dataset:

    elif 'liauto' in dataset:
        return 'LiAuto'
    else:
        # if dataset not in dataset_URLs:
        #     import warnings
        #     warnings.warn(f"Dataset {dataset} not found in dataset_URLs, will use 'multi-choice' as the default TYPE.")
        #     return 'multi-choice'
        # else:
        #     return 'QA'
        import warnings
        warnings.warn(f"Dataset {dataset} not found, will use 'VQA' as the default TYPE.")
        return 'VQA'

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # work_dir
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        save_dir = args.work_dir
    else:
        # use config filename as default work_dir
        save_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.data_path))[0])
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    # save_dir = osp.join(save_dir, timestamp)
    mkdir_or_exist(osp.abspath(save_dir))
    print('=======================================================')
    print(f'Dataset path: {osp.abspath(args.data_path)}\n'
          f'Results will be saved to {osp.abspath(save_dir)}')
    print('=======================================================')
    bench_name = args.data_path.split("/")[-1].replace(".tsv", "")
    results_xlsx_path = osp.join(save_dir, f"{save_dir.split('/')[-1]}_{bench_name}.xlsx")
    print(f"results_xlsx_path: {results_xlsx_path}")
    results_json_path = osp.join(save_dir, f'{bench_name}_result.json')
    args_path = osp.join(save_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # # ====> load from pth
    # from xtuner.configs.llava.yi_family_clip_vit_large_p14_336.evaluate.llava_nh_yi_34b_qlora_eva_vit_large_p14_336_lora_e1_gpu8_finetune_multisubimages import model as model_dict

    # model = BUILDER.build(model_dict)

    # model.cuda()
    # model.eval()
    # # <==== load from pth end


    # print(model.visual_encoder)

    # for n, p in model.llm.named_parameters():
    #     print(n, p.mean(), p.max(), p.dtype)
    #     torch.save(p, f'/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/tensors/{n}.tensor')

    # build llm
    quantization_config = None
    load_in_8bit = False
    if args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    elif args.bits == 8:
        load_in_8bit = True
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': 'auto',
        'offload_folder': args.offload_folder,
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
    }

    # build llm
    # llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
    #                                            **model_kwargs)
    
    llm = LlamaForCausalLM_VoteMix.from_pretrained(args.model_name_or_path,
                                               **model_kwargs)

    # llm = LlamaForCausalLM_PDrop.from_pretrained(args.model_name_or_path,
    #                                            **model_kwargs)

    llm_name = llm.__class__.__name__

    print(f"llm_name: {llm_name}")

    if 'mindgpt' in args.model_name_or_path.lower() or 'mind-gpt' in args.model_name_or_path.lower():
        print(f"Use tiktoken for {args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/work_dirs_guan/xtuner_ckpt_validation_1",
            trust_remote_code=True,
            encode_special_tokens=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            encode_special_tokens=True)
    print(f'Load LLM from {args.model_name_or_path}')

    llava_path = snapshot_download(
        repo_id=args.llava) if not osp.isdir(args.llava) else args.llava
    
    # visual_encoder = model.visual_encoder

    pos_embed_replace = True

    # build visual_encoder
    if 'visual_encoder' in os.listdir(llava_path):
        assert args.visual_encoder is None, (
            "Please don't specify the `--visual-encoder` since passed "
            '`--llava` contains a visual encoder!')
        visual_encoder_path = osp.join(llava_path, 'visual_encoder')
        pos_embed_replace = False
    else:
        assert args.visual_encoder is not None, (
            'Please specify the `--visual-encoder`!')
        visual_encoder_path = args.visual_encoder


    if 'convnext_visual_encoder' in os.listdir(llava_path):
        assert args.convnext_visual_encoder is None, (
            "Please don't specify the `--convnext_visual_encoder` since passed "
            '`--llava` contains a visual encoder!')
        convnext_visual_encoder_path = osp.join(llava_path, 'convnext_visual_encoder')
    else:
        # assert args.convnext_visual_encoder is not None, (
        #     'Please specify the `--convnext_visual_encoder`!')
        convnext_visual_encoder_path = args.convnext_visual_encoder


    if args.dino_visual_encoder is not None:
        dino_visual_encoder_path = args.dino_visual_encoder
    else:
        dino_visual_encoder_path = None


    dynamic_image_size = args.dynamic_image_size
    convnext_image_size= args.convnext_image_size

    from xtuner.model.llava import resample_abs_pos_embed, interpolate_pos_embed


    # if 'cjy'  in visual_encoder_path:
    #     print("use mindvit")

    #     if dynamic_image_size is not None:
    #         hf_config = AutoConfig.from_pretrained(visual_encoder_path, trust_remote_code=True)
    #         hf_config.dynamic_image_size = dynamic_image_size

    #         print(f'interpolate pos embed for mindvit to {dynamic_image_size}')

    #         visual_encoder = AutoModel.from_pretrained(config=hf_config,
    #                                                     pretrained_model_name_or_path=visual_encoder_path,
    #                                                     torch_dtype=torch.float16,
    #                                                     device_map='cpu',
    #                                                     trust_remote_code=True, 
    #                                                     ignore_mismatched_sizes=True)

    #         print(f'resample_abs_pos_embed for mindvit eva automodel')

    #         position_embedding_weight = resample_abs_pos_embed(
    #             visual_encoder.vision_model.embeddings.position_embedding.weight.unsqueeze(0),
    #             (dynamic_image_size // visual_encoder.vision_model.embeddings.patch_size, dynamic_image_size // visual_encoder.vision_model.embeddings.patch_size),
    #             num_prefix_tokens=visual_encoder.vision_model.embeddings.num_prefix_tokens,
    #         ).squeeze(0)     
    #     else:
    #         hf_config = AutoConfig.from_pretrained(visual_encoder_path, trust_remote_code=True)
    #         visual_encoder = AutoModel.from_pretrained(visual_encoder_path, config=hf_config, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, 
    #                                                 ignore_mismatched_sizes=True)

    if 'siglip' in visual_encoder_path.lower():

        print("use siglip vit")

        # try:
        visual_encoder = SiglipVisionModel.from_pretrained(visual_encoder_path, 
                                                            # torch_dtype=torch.float16, 
                                                            torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],
                                                            device_map='cpu')

        if dynamic_image_size is not None and pos_embed_replace:
            print(f'interpolate pos embed for siglip to {dynamic_image_size}')

            pos_embed = visual_encoder.vision_model.embeddings.position_embedding.weight
            print(f"pos_embed: {pos_embed.shape}")
            pos_embed = pos_embed.float()
            embedding_size = pos_embed.shape[-1]
            orig_size = int(visual_encoder.vision_model.embeddings.image_size / visual_encoder.vision_model.embeddings.patch_size)
            new_size = int(dynamic_image_size / visual_encoder.vision_model.embeddings.patch_size)
            
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))

            pos_embed = pos_embed.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(
                pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
            position_embedding_weight = pos_embed.permute(0, 2, 3, 1).flatten(1, 2).squeeze()
            position_embedding_weight = position_embedding_weight.to(visual_encoder.dtype)
            print(f"position_embedding_weight: {position_embedding_weight.shape}")
    elif 'radio' in visual_encoder_path.lower():
        print("use radio vit")
        visual_encoder=AutoModel.from_pretrained(visual_encoder_path, trust_remote_code=True)
    else: 
        print("use open clip vit")
        visual_encoder = CLIPVisionModel.from_pretrained(visual_encoder_path, 
                                                        # torch_dtype=torch.float16, 
                                                        torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],
                                                        device_map='cpu')

        if dynamic_image_size is not None:
            print(f'interpolate pos embed for clip to {dynamic_image_size}')
            pos_embed_replace = True
            position_embedding_weight = interpolate_pos_embed(visual_encoder, dynamic_image_size)

    if dynamic_image_size is not None and pos_embed_replace:
        num_patches = (dynamic_image_size // visual_encoder.vision_model.embeddings.patch_size) ** 2
        if visual_encoder.__class__.__name__ == 'SiglipVisionModel':
            print(f'replace pos embed for SiglipVisionModel', 'current')
            num_positions = num_patches
        else:
            num_positions = num_patches + 1
        position_embedding = torch.nn.Embedding(num_positions, visual_encoder.vision_model.embeddings.embed_dim)
        visual_encoder.vision_model.embeddings.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent = False)
        position_embedding.weight = torch.nn.Parameter(position_embedding_weight)
        visual_encoder.vision_model.embeddings.position_embedding = position_embedding

    print(f'Load visual_encoder from {visual_encoder_path}')

    if dino_visual_encoder_path is not None:
        hf_config = AutoConfig.from_pretrained(dino_visual_encoder_path, trust_remote_code=True)

        dino_visual_encoder = AutoModel.from_pretrained(
            config=hf_config,
            pretrained_model_name_or_path=dino_visual_encoder_path,
            # torch_dtype=torch.float16,
            torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],
            device_map='cpu',
            trust_remote_code=True, 
            # ignore_mismatched_sizes=True
        )
        print(f'Load dino_visual_encoder from {dino_visual_encoder_path}')
    else:
        dino_visual_encoder = None


    if convnext_visual_encoder_path is not None:
        # import open_clip
        # convnext_visual_encoder = open_clip.create_model(convnext_visual_encoder_path[0], pretrained=convnext_visual_encoder_path[1]).visual.trunk
        # convnext_visual_encoder = ConvNextModel.from_pretrained(
        #     pretrained_model_name_or_path=convnext_visual_encoder_path,
        #     trust_remote_code=True,
        #     torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],
        #     device_map='cpu'
        # )
        convnext_visual_encoder = timm.create_model(
                        model_name='convnext_xxlarge.clip_laion2b_soup_ft_in1k',
                        pretrained=True,
                        # features_only=True,
                        pretrained_cfg_overlay=dict(file=convnext_visual_encoder_path))
        
        print(f'Load convnext_visual_encoder from {convnext_visual_encoder_path}')
    else:
        convnext_visual_encoder = None

    # dino_image_processor = None
    # if dynamic_image_size is not None:
    #     image_processor = CLIPImageProcessor.from_pretrained(visual_encoder_path, crop_size=dynamic_image_size, size=dynamic_image_size)
    #     # if dino_visual_encoder_path is not None:
    #     #     dino_image_processor = CLIPImageProcessor.from_pretrained(dino_visual_encoder_path, crop_size=dynamic_image_size, size=dynamic_image_size)
    # elif visual_encoder.__class__.__name__ == 'SiglipVisionModel':
    #     image_processor = CLIPImageProcessor.from_pretrained(visual_encoder_path, crop_size=384, size=384)
    #     # if dino_visual_encoder_path is not None:
    #     #     dino_image_processor = CLIPImageProcessor.from_pretrained(dino_visual_encoder_path, crop_size=384, size=384)
    # else:
    #     if visual_encoder.__class__.__name__ == 'RADIOModel':
    #         image_processor = CLIPImageProcessor.from_pretrained(
    #             pretrained_model_name_or_path=visual_encoder_path,
    #             trust_remote_code=True,
    #             do_resize=True,
    #             do_center_crop=True,
    #             crop_size=336, 
    #             size=336
    #         )           
    #     else:
    #         image_processor = CLIPImageProcessor.from_pretrained(visual_encoder_path)

    convnext_image_processor = None
    if convnext_image_size is not None:
        convnext_image_processor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path='/mnt/pfs-mc0p4k/cv/team/sunhaoyi/xtuner-ved/hf_hub/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup',
            trust_remote_code=True,
            crop_size=convnext_image_size, 
            size=convnext_image_size
        )
        print(f'convnext_image_processor: {convnext_image_processor}')

    # print(f'image_processor: {image_processor}')
    # print(f"dino_image_processor: {dino_image_processor}")
    # print(f"convnext_image_processor: {convnext_image_processor}")

    dynamic_image_processor = CLIPImageProcessor.from_pretrained(visual_encoder_path, crop_size=dynamic_image_size, size=dynamic_image_size)

    # from xtuner.utils import Dynamic_CV_CLIPImageProcessor
    # dynamic_image_processor = Dynamic_CV_CLIPImageProcessor.from_pretrained(
    #     pretrained_model_name_or_path=visual_encoder_path,
    #     trust_remote_code=True, 
    #     do_resize=True,
    #     # do_center_crop=True
    # )
    
    square_image_processor = CLIPImageProcessor.from_pretrained(visual_encoder_path, crop_size=dynamic_image_size, size=dynamic_image_size)
    # square_image_processor = None 

    print(f'dynamic_image_processor: {dynamic_image_processor}')
    print(f'square_image_processor: {square_image_processor}')

    visual_encoder_name = visual_encoder.__class__.__name__

    # load adapter
    if 'llm_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'llm_adapter')
        llm = PeftModel.from_pretrained(
            llm,
            adapter_path,
            offload_folder=args.offload_folder,
            trust_remote_code=True)
        print(f'Load LLM adapter from {adapter_path}')

    if 'visual_encoder_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'visual_encoder_adapter')
        visual_encoder = PeftModel.from_pretrained(
            visual_encoder,
            adapter_path,
            offload_folder=args.offload_folder,
            trust_remote_code=True)
        print(f'Load visual_encoder adapter from {adapter_path}')

        # visual_encoder = visual_encoder.merge_and_unload()
        # save_dir = '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/work_dirs/llava_mindgpt_moe_siglipdino_448_e1_gpu8_pretrain/mindgpt-moe-chat-siglipdino-558k-tiktoken_iter_437_hf/siglip-so400m-patch14-384-with-lora'
        # print(f'Saving to {save_dir}...')
        # visual_encoder.save_pretrained(
        #     save_dir,
        #     safe_serialization=False,
        #     # max_shard_size=args.max_shard_size
        # )

    if 'dino_visual_encoder_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'dino_visual_encoder_adapter')
        dino_visual_encoder = PeftModel.from_pretrained(dino_visual_encoder,
                                                        adapter_path,
                                                        offload_folder=args.offload_folder,
                                                        trust_remote_code=True,
                                                        # device_map='cpu'
                                                        )
        print(f'Load dino_visual_encoder_adapter adapter from {adapter_path}')

    if 'convnext_visual_encoder_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'convnext_visual_encoder_adapter')
        convnext_visual_encoder = PeftModel.from_pretrained(convnext_visual_encoder,
                                                            adapter_path,
                                                            offload_folder=args.offload_folder,
                                                            trust_remote_code=True,
                                                            # device_map='cpu'
                                                            )
        print(f'Load convnext_visual_encoder_adapter adapter from {adapter_path}')

    # build projector
    projector_path = osp.join(llava_path, 'projector')
    print(f"projector_path: {projector_path}")
    projector = AutoModel.from_pretrained(
        projector_path,
        torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],
        trust_remote_code=True)
    print(f"projector arch: {projector}")
    print(f'Load projector from {projector_path}')

    projector.cuda()
    projector.eval()
    visual_encoder.cuda()
    visual_encoder.eval()
    if dino_visual_encoder:
        dino_visual_encoder.cuda()
        dino_visual_encoder.eval()
    if convnext_visual_encoder:
        convnext_visual_encoder.cuda()
        convnext_visual_encoder.eval()
    llm.eval()

    num_sub_images = args.num_sub_images
    if num_sub_images > 0:
            print(f"inference with multi subimages")

    stop_words = args.stop_words
    if args.prompt_template:
        template = PROMPT_TEMPLATE[args.prompt_template]
        stop_words += template.get('STOP_WORDS', [])
    stop_criteria = get_stop_criteria(
        tokenizer=tokenizer, stop_words=stop_words)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    bench_type = DATASET_TYPE(bench_name)
    print(f"Benchmark [{bench_name}] Type: {bench_type}", flush=True)
    dataset = MMBenchDataset(args.data_path, bench_type)
    results = []
    n_samples = len(dataset)
    for i in tqdm.tqdm(range(n_samples)):
        
        data_sample = dataset[i]

        if bench_type == 'multi-choice':
            if 'MME-RealWorld' in bench_name:
                question = data_sample['question']
                choice_prompt = data_sample['multi-choice options'] + '\n'
                SYS = {
                    'MME-RealWorld': (
                        'Select the best answer to the above multiple-choice question based on the image. '
                        'Respond with only the letter (A, B, C, D, or E) of the correct option. \n'
                        'The best answer is:'
                    ),
                    'MME-RealWorld-CN': (
                        '根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n'
                        '最佳答案为：'
                    ),
                }

                question += ' ' + choice_prompt + SYS[bench_name]
                text = question
            else:
                question = data_sample['question']
                text = ''
                if data_sample['context'] is not None:
                    text += 'Hint: {}\n'.format(data_sample['context'])
                text += f'Question: {question}\n'
                text += 'Options:\n' + data_sample['options']
                if is_cn_string(text):
                    text = text + '请直接回答选项字母。'
                else:
                    text = text + ("Answer with the option's letter from the "
                                'given choices directly.')
                # text += 'Please select the correct answer from the options above. \n'
        elif bench_type == 'TextVQA':
            question = data_sample['question']
            text = ''
            if data_sample['context'] is not None:
                text += 'Hint: {}\n'.format(data_sample['context'])
            text += f'{question}\n'
            # text += f'Question: {question}\n'
            # text += 'Options:\n' + data_sample['options']
            if is_cn_string(text):
                text = text + '请直接回答选项字母。'
            else:
                text = text + ('Answer the question using a single word or phrase.')
        elif bench_type == 'LiAuto':
            assert data_sample['type'] in ['choice', 'bool']
            if data_sample['type'] == 'choice':
                question = data_sample['question']
                text = ''
                if data_sample['context'] is not None:
                    text += 'Hint: {}\n'.format(data_sample['context'])
                text += f'Question: {question}\n'
                text += 'Options:\n' + data_sample['options']
                if is_cn_string(text):
                    text = text + '请直接回答选项字母。'
                else:
                    text = text + ("Answer with the option's letter from the "
                                'given choices directly.')
            elif data_sample['type'] == 'bool':
                text = data_sample['question']
        else:
            text = data_sample['question']

        # text = "Please describe this picture."

        text = DEFAULT_IMAGE_TOKEN + '\n' + text
        if args.prompt_template:
            prompt_text = ''
            template = PROMPT_TEMPLATE[args.prompt_template]
            prompt_text += template['INSTRUCTION'].format(
                input=text, round=1, bot_name=args.bot_name)

            if template.get('TYPE', '') and template['TYPE'] == 'mindgpt':
                # system = template['SYSTEM'].format(system='你是一个名字叫做理想同学的AI数字生命体。')
                # system = template['SYSTEM'].format(system='你是一个名字叫做理想同学的AI数字生命体。\n理想同学是一个可靠的智能家庭助手，由理想汽车智能空间部门创造。\n理想同学能解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、中立的、安全的回复。\n\n请根据以下文本写一个合适的回复。')
                system = template['SYSTEM'].format(system='你是由理想汽车智能空间创造的多模态AI助手，名叫理想同学，拥有处理和分析图像的能力。\n你的主要任务是根据用户提供的信息提供准确、有用的回复。')
                prompt_text = system + prompt_text
            
            # system = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
            # if system != '' and system is not None:
            #     system = template.SYSTEM.format(system=system)
            #     prompt_text = system + prompt_text
        else:
            prompt_text = text
        inputs = prompt_text
        print(f"inputs: {[inputs]}", flush=True)

        # inputs = inputs.replace('<|im_start|>', '[unused0]').replace('<|im_end|>', '[unused1]')
        # print(f"inputs: {[inputs]}")

        if data_sample['img'] is not None:
            image = data_sample['img']

            # image = load_image('/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/view.jpg')
            # image = load_image('/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/view_1080p.jpg')

            print(f"image: {image.shape}")

            if dynamic_image_processor is not None:

                try:
                    input_size = dynamic_image_processor.crop_size['height']
                except Exception as e:
                    input_size = dynamic_image_processor.crop_size

                # input_size = 640
                
                # ===> internvl split tiles
                # image_thumbnail = dynamic_image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                # num_pixel_values = 1

                # ===> minimonkey split tiles
                image_thumbnail, _ = expand_original_aspect(image, image_size=input_size, background_color=tuple(int(x * 255) for x in dynamic_image_processor.image_mean))
                image_thumbnail = dynamic_image_processor.preprocess(image_thumbnail, return_tensors='pt')['pixel_values']

                # image_thumbnail = dynamic_image_processor.preprocess(image, return_tensors='pt')['pixel_values']

                image_thumbnail = image_thumbnail.cuda()
                print(f"image_thumbnail: {image_thumbnail.shape}", flush=True)
                 
                if convnext_image_processor is not None:
                    convnext_image_thumbnail = convnext_image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                    convnext_image_thumbnail = convnext_image_thumbnail.cuda()
                    print(f"convnext_image_thumbnail: {convnext_image_thumbnail.shape}", flush=True)
                else:
                    convnext_image_thumbnail = None

                # ===> naive resolution
                # image_thumbnail = dynamic_image_processor.preprocess(image, return_tensors='pt')['pixel_values']

                num_pixel_values = 1
                image_size = [image_thumbnail.shape[2], image_thumbnail.shape[3]]
            else:
                image_thumbnail = None
                convnext_image_thumbnail = None

            if  square_image_processor is not None:

                if convnext_image_processor is not None:
                    try:
                        input_size = convnext_image_processor.crop_size['height']
                    except Exception as e:
                        input_size = convnext_image_processor.crop_size        
                else:
                    try:
                        input_size = square_image_processor.crop_size['height']
                    except Exception as e:
                        input_size = square_image_processor.crop_size  
                
                # input_size = 640

                print(f"use input_size {input_size} for slidingwindow and minimonkey ")      

                # ===> internvl split tiles
                # image_tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=12, min_num=4)
                # num_pixel_values += len(image_tiles)
                # image_tiles = square_image_processor.preprocess(image_tiles, return_tensors='pt')['pixel_values']

                # ===> minimokey split tiles
                sw_image_tiles = slidingwindow_preprocess(image, image_size=input_size, background_color=tuple(int(x * 255) for x in square_image_processor.image_mean))
                minimokey_image_tiles = dynamic_preprocess(image, min_num=3, max_num=7, image_size=input_size, use_rounded_down_ratio=True)
                # minimokey_image_tiles = dynamic_preprocess(image, min_num=2, max_num=6, image_size=input_size, use_rounded_down_ratio=True)
                if sw_image_tiles is None:
                    num_pixel_values += len(minimokey_image_tiles)
                    image_tiles2 = square_image_processor.preprocess(minimokey_image_tiles, return_tensors='pt')['pixel_values']
                    image_tiles = image_tiles2
                    image_tiles = image_tiles.cuda()
                    print(f"minimokey image_tiles: {image_tiles.shape}")
                    if convnext_image_processor is not None:
                        convnext_image_tiles2 = convnext_image_processor.preprocess(minimokey_image_tiles, return_tensors='pt')['pixel_values']
                        convnext_image_tiles = convnext_image_tiles2
                        convnext_image_tiles = convnext_image_tiles.cuda()
                        print(f"minimokey convnext_image_tiles: {convnext_image_tiles.shape}")
                    else:
                        convnext_image_tiles = None 
                else:
                    num_pixel_values += (len(sw_image_tiles) + len(minimokey_image_tiles))
                    image_tiles1 = square_image_processor.preprocess(sw_image_tiles, return_tensors='pt')['pixel_values']
                    image_tiles2 = square_image_processor.preprocess(minimokey_image_tiles, return_tensors='pt')['pixel_values']
                    image_tiles = torch.cat([image_tiles1, image_tiles2], dim=0)
                    image_tiles = image_tiles.cuda()
                    print(f"sw + minimokey ({len(sw_image_tiles)} + {len(minimokey_image_tiles)}) image_tiles: {image_tiles.shape}")
                    if convnext_image_processor is not None:
                        convnext_image_tiles1 = convnext_image_processor.preprocess(sw_image_tiles, return_tensors='pt')['pixel_values']
                        convnext_image_tiles2 = convnext_image_processor.preprocess(minimokey_image_tiles, return_tensors='pt')['pixel_values']
                        convnext_image_tiles = torch.cat([convnext_image_tiles1, convnext_image_tiles2], dim=0)
                        convnext_image_tiles = convnext_image_tiles.cuda()
                        print(f"sw + minimokey ({len(sw_image_tiles)} + {len(minimokey_image_tiles)}) convnext_image_tiles: {convnext_image_tiles.shape}")
                    else:
                        convnext_image_tiles = None
            else:
                image_tiles = None
                convnext_image_tiles = None
                    
                # images = dynamic_preprocess(image, image_size=input_size,  max_num=num_sub_images, use_thumbnail=True, 
                #                             thumbnail_first=True
                #                             )
                # num_pixel_values = len(images)
                # print(f"num_pixel_values: {num_pixel_values}")
                # image = image_processor.preprocess(images, return_tensors='pt')['pixel_values']

                # num_padding = num_sub_images + 1 - num_pixel_values
                # if num_padding > 0:
                #     image = torch.cat([image, torch.zeros(num_padding, 3, input_size, input_size)], dim=0)

                # image = image.cuda()

                # if dino_image_processor:
                #     image_dino = dino_image_processor.preprocess(images, return_tensors='pt')['pixel_values']
                #     image_dino = image_dino.cuda()
                # else:
                #     image_dino = image

                # if convnext_image_processor:
                #     image_convnext = convnext_image_processor(images, return_tensors='pt')['pixel_values']
                #     image_convnext = image_convnext.cuda()
                # else:
                #     image_convnext = None

            # else:
            #     if image_processor.image_mean is not None:
            #         square_image = expand2square(
            #             image,
            #             tuple(int(x * 255) for x in image_processor.image_mean))
            #     else:
            #         square_image = image

            #     image = image_processor.preprocess(
            #         square_image, return_tensors='pt')['pixel_values'][0]
            #     image = image.cuda().unsqueeze(0)

            #     if dino_image_processor:
            #         image_dino = image_processor.preprocess(
            #             square_image, return_tensors='pt')['pixel_values'][0]
            #         image_dino = image_dino.cuda().unsqueeze(0)
            #     else:
            #         image_dino = image

            #     if convnext_image_processor:
            #         image_convnext = convnext_image_processor(
            #             square_image, return_tensors='pt')['pixel_values']
            #         image_convnext = image_convnext.cuda()
            #     else:
            #         image_convnext = None
                    
            # print(f"image: {image.shape}")
            # if image_dino is not None:
            #     print(f"image_dino: {image_dino.shape}, copy from image: {(image-image_dino).mean()}")
            # if image_convnext is not None:
            #     print(f"image_convnext: {image_convnext.shape}")

            with torch.no_grad():

                bs = image_thumbnail.shape[0]

                if (image_thumbnail is not None) and (image_tiles is not None):
                    print(f"cat thumbnail and tiles")
                    image = torch.cat([image_thumbnail, image_tiles], dim=0)

                    if (convnext_image_thumbnail is not None) and (convnext_image_tiles is not None):
                        convnext_image = torch.cat([convnext_image_thumbnail, convnext_image_tiles], dim=0)
                else:
                    if image_thumbnail is not None:
                        print(f"only use thumbnail feature")
                        image = image_thumbnail
                    if image_tiles is not None:
                        print(f"only use tiles feature")
                        image = image_tiles

                image = image.to(visual_encoder.dtype)
                if visual_encoder_name == "RADIOModel":
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        visual_outputs = visual_encoder(image)
                else:
                    visual_outputs = visual_encoder(image, output_hidden_states=True)

                if dino_visual_encoder:
                    image = image.to(dino_visual_encoder.dtype)
                    dino_outputs = dino_visual_encoder(image, output_hidden_states=True)

                if convnext_visual_encoder:
                    # image_convnext = image_convnext.to(convnext_visual_encoder.dtype)
                    # convnext_outputs = convnext_visual_encoder(image_convnext, output_hidden_states=True).last_hidden_state

                    convnext_image = convnext_image.to(convnext_visual_encoder.stem[0].weight.dtype)

                    convnext_outputs = convnext_visual_encoder.stem(convnext_image)
                    for stage in convnext_visual_encoder.stages:
                        convnext_outputs = stage(convnext_outputs)

                visual_feat_list = []
                if visual_encoder_name == "RADIOModel":
                    vit_pixel_values = visual_outputs[-1]
                    visual_feat_list.append(vit_pixel_values)
                else:
                    vit_pixel_values = visual_outputs.hidden_states[args.visual_select_layer]
                    visual_feat_list.append(vit_pixel_values)

                if dino_visual_encoder:
                    # print(f"dino_visual_encoder")
                    dino_pixel_values = dino_outputs.hidden_states[args.visual_select_layer][:, 1:]
                    visual_feat_list.append(dino_pixel_values)

                if convnext_visual_encoder:
                    # print(f"convnext_visual_encoder")
                    # bs = vit_pixel_values.shape[0]
                    # convnext_pixel_values = convnext_outputs.view(bs, convnext_outputs.shape[1], -1).permute(0, 2, 1)
                    convnext_pixel_values = rearrange(convnext_outputs, "b d h w -> b (h w) d")
                    visual_feat_list.append(convnext_pixel_values)
                
                if args.fusion_channel:
                    pixel_values = projector.vit_channel_fusion([x.to(projector.dtype) for x in visual_feat_list])
                else:
                    pixel_values = torch.cat(visual_feat_list, dim=-1).to(projector.dtype)

                print(f"pixel_values: {pixel_values.shape}")

                if (image_thumbnail is not None) and (image_tiles is not None):
                    thumbnail_pixel_values = pixel_values[:bs, ...]
                    tiles_pixel_values = pixel_values[bs:, ...]

                    visual_feat_list = [thumbnail_pixel_values, tiles_pixel_values]

                # visual_feat_list = [pixel_values]

                if len(visual_feat_list) > 1:
                    pixel_values = projector(visual_feat_list).to(llm.dtype)
                else:
                    pixel_values = projector(
                        visual_feat_list[0].to(projector.dtype)
                        # (visual_feat_list[0].to(projector.dtype), (x // visual_encoder.patch_size for x in image_size))
                    ).to(llm.dtype)

                print(f"pixel_values: {pixel_values.shape}", flush=True)

                chunk_encode = []
                for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
                    # print(f"chunk: {[chunk]}")
                    if idx == 0:
                        cur_encode = tokenizer.encode(chunk)
                    else:
                        cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
                    chunk_encode.append(cur_encode)
                assert len(chunk_encode) == 2
                ids = []
                for idx, cur_chunk_encode in enumerate(chunk_encode):
                    ids.extend(cur_chunk_encode)
                    if idx != len(chunk_encode) - 1:
                        ids.append(IMAGE_TOKEN_INDEX)
                ids = torch.tensor(ids).cuda().unsqueeze(0)

                # print(f"ids: {ids}")
                # quit()

                vote_and_mix = False
                add_img_slice_special_tokens = True

                if (image_thumbnail is not None) and (image_tiles is not None) and (not vote_and_mix):

                    if len(args.projector_num_queries) == 1:
                        pixel_values = pixel_values.view(1, -1, pixel_values.shape[2])
                    # print(f"pixel_values: {pixel_values.shape}")
                    
                    # thumbnail_pixel_values = pixel_values[:bs, ...]
                    # tiles_pixel_values = pixel_values[bs:, ...].view(bs, -1, pixel_values.shape[-1])
                    # pixel_values = torch.cat([thumbnail_pixel_values, tiles_pixel_values], dim=1)
                    
                    print(f"num_pixel_values: {num_pixel_values}", flush=True)

                    print(f"pixel_values to prepare_multi_subimages_inputs_labels_for_multimodal: {pixel_values.shape}", flush=True)

                    mm_inputs = prepare_multi_subimages_inputs_labels_for_multimodal(
                        llm=llm,
                        llm_name=llm_name,
                        input_ids=ids,
                        pixel_values=pixel_values, 
                        num_pixel_values=[num_pixel_values],
                        projector_num_queries=args.projector_num_queries, 
                        add_img_slice_special_tokens=add_img_slice_special_tokens, 
                        mark_slice_indices=True
                    )
                else:
                    print(f"pixel_values to prepare_inputs_labels_for_multimodal: {pixel_values.shape}", flush=True)
                    mm_inputs = prepare_inputs_labels_for_multimodal(
                        llm=llm, input_ids=ids, pixel_values=pixel_values)

                generate_output = llm.generate(
                    **mm_inputs,
                    generation_config=gen_config,
                    streamer=None,
                    bos_token_id=tokenizer.bos_token_id,
                    stopping_criteria=stop_criteria
                    )
                
                # print(f"generate_output[0]: {generate_output[0]}")

                predict = tokenizer.decode(
                    generate_output[0], skip_special_tokens=True).strip()
                
                if bench_type == "multi-choice":
                    predict = predict.replace('Answer: ', '').replace('Answer', '').replace(': ', '').replace('回答', '').replace('答案', '').replace('：', '')

                print(f"predict: {[predict]}", flush=True)
                # quit()

            cur_result = {}
            cur_result['question'] = data_sample.get('question')
            cur_result.update(data_sample.get('options_dict'))
            cur_result['prediction'] = predict
            if data_sample.get('conversations') is not None:
                cur_result['conversations'] = data_sample.get('conversations')
            if data_sample.get('category') is not None:
                cur_result['category'] = data_sample.get('category')
            if data_sample.get('l2-category') is not None:
                cur_result['l2-category'] = data_sample.get('l2-category')
            cur_result['index'] = data_sample.get('index')
            cur_result['split'] = data_sample.get('split')
            cur_result['answer'] = data_sample.get('answer')
            if data_sample.get('width') is not None:
                cur_result['width'] = data_sample.get('width') 
            if data_sample.get('height') is not None:
                cur_result['height'] = data_sample.get('height')
            results.append(cur_result)
        else:
            cur_result = {}
            cur_result['question'] = data_sample.get('question')
            cur_result.update(data_sample.get('options_dict'))
            cur_result['prediction'] = ''
            if data_sample.get('category') is not None:
                cur_result['category'] = data_sample.get('category')
            if data_sample.get('l2-category') is not None:
                cur_result['l2-category'] = data_sample.get('l2-category')
            cur_result['index'] = data_sample.get('index')
            cur_result['split'] = data_sample.get('split')
            cur_result['answer'] = data_sample.get('answer')
            results.append(cur_result)
                    
    results_df = pd.DataFrame(results)
    with pd.ExcelWriter(results_xlsx_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, index=False)

    if dataset.split == 'dev':
        if bench_type == 'LiAuto':
            results_dict = dataset.eval_li_result(results_df, show=True)
            with open(results_json_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
        elif bench_type == 'multi-choice':
            results_dict = dataset.eval_result(results_df, show=True)
            with open(results_json_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
        elif bench_type == 'Y/N':
            results_dict = dataset.eval_multi_choice(results_df, show=True)
            with open(results_json_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
        else:
            print('All done!')
    else:
        print('All done!')

if __name__ == '__main__':
    main()
