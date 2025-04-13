import argparse
import os
import os.path as osp
import re
import sys
import re
import cv2
import numpy as np
from typing import List, Optional

import gradio as gr

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (AutoModel, AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, 
                        #   CLIPImageProcessor, 
                          CLIPVisionModel, 
                          SiglipImageProcessor, SiglipVisionModel,
                          GenerationConfig)

from xtuner.utils import CV_CLIPImageProcessor as CLIPImageProcessor

from xtuner.dataset.utils import expand2square, load_image
from xtuner.dataset.llava_multisubimages import dynamic_preprocess
from xtuner.model.utils import prepare_inputs_labels_for_multimodal, prepare_multi_subimages_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria, get_streamer
from xtuner.utils import (IGNORE_INDEX,
                          IMAGE_TOKEN_INDEX,
                          DEFAULT_IMAGE_TOKEN,
                          PROMPT_TEMPLATE, 
                          SYSTEM_TEMPLATE)

# DEFAULT_IMAGE_TOKEN = '[unused10]'
# IMAGE_TOKEN_INDEX = 94886

n_turn = 0
inputs, format_inputs = '', ''
last_turn_image_input = None

# def prepare_inputs_labels_for_multimodal(
#         llm: PreTrainedModel,
#         input_ids: torch.LongTensor = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         labels: Optional[torch.LongTensor] = None,
#         pixel_values: Optional[torch.FloatTensor] = None):
#     if pixel_values is None:
#         # print(f"====> pixel_values is None")
#         return {
#             'input_ids': input_ids,
#             'position_ids': position_ids,
#             'attention_mask': attention_mask,
#             'past_key_values': past_key_values,
#             'inputs_embeds': None,
#             'labels': labels
#         }

#     _labels = labels
#     _position_ids = position_ids
#     _attention_mask = attention_mask
#     if attention_mask is None:
#         attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
#     else:
#         attention_mask = attention_mask.bool()
#     if position_ids is None:
#         position_ids = torch.arange(
#             0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
#     if labels is None:
#         labels = torch.full_like(input_ids, IGNORE_INDEX)

#     # remove the padding using attention_mask -- TODO: double check
#     input_ids = [
#         cur_input_ids[cur_attention_mask]
#         for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
#     ]
#     labels = [
#         cur_labels[cur_attention_mask]
#         for cur_labels, cur_attention_mask in zip(labels, attention_mask)
#     ]

#     new_inputs_embeds = []
#     new_labels = []
#     cur_image_idx = 0
#     for batch_idx, cur_input_ids in enumerate(input_ids):
#         num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
#         if num_images == 0:
#             cur_pixel_values = pixel_values[cur_image_idx]
#             cur_inputs_embeds_1 = llm.get_input_embeddings()(cur_input_ids)
#             cur_inputs_embeds = torch.cat(
#                 [cur_inputs_embeds_1, cur_pixel_values[0:0]], dim=0)
#             new_inputs_embeds.append(cur_inputs_embeds)
#             new_labels.append(labels[batch_idx])
#             cur_image_idx += 1
#             continue

#         image_token_indices = [-1] + torch.where(
#             cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
#                 cur_input_ids.shape[0]
#             ]
#         cur_input_ids_noim = []
#         cur_labels = labels[batch_idx]
#         cur_labels_noim = []
#         for i in range(len(image_token_indices) - 1):
#             cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
#             cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
#         split_sizes = [x.shape[0] for x in cur_labels_noim]
#         cur_inputs_embeds = llm.get_input_embeddings()(
#             torch.cat(cur_input_ids_noim))
#         cur_inputs_embeds_no_im = torch.split(
#             cur_inputs_embeds, split_sizes, dim=0)
#         cur_new_inputs_embeds = []
#         cur_new_labels = []

#         for i in range(num_images + 1):
#             cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
#             cur_new_labels.append(cur_labels_noim[i])
#             if i < num_images:
#                 cur_pixel_values = pixel_values[cur_image_idx]
#                 cur_image_idx += 1
#                 cur_new_inputs_embeds.append(cur_pixel_values)
#                 cur_new_labels.append(
#                     torch.full((cur_pixel_values.shape[0], ),
#                                IGNORE_INDEX,
#                                device=cur_labels.device,
#                                dtype=cur_labels.dtype))

#         cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
#         cur_new_labels = torch.cat(cur_new_labels)

#         new_inputs_embeds.append(cur_new_inputs_embeds)
#         new_labels.append(cur_new_labels)

#     # Combine them
#     max_len = max(x.shape[0] for x in new_inputs_embeds)
#     batch_size = len(new_inputs_embeds)

#     new_inputs_embeds_padded = []
#     new_labels_padded = torch.full((batch_size, max_len),
#                                    IGNORE_INDEX,
#                                    dtype=new_labels[0].dtype,
#                                    device=new_labels[0].device)
#     attention_mask = torch.zeros((batch_size, max_len),
#                                  dtype=attention_mask.dtype,
#                                  device=attention_mask.device)
#     position_ids = torch.zeros((batch_size, max_len),
#                                dtype=position_ids.dtype,
#                                device=position_ids.device)

#     for i, (cur_new_embed,
#             cur_new_labels) in enumerate(zip(new_inputs_embeds, new_labels)):
#         cur_len = cur_new_embed.shape[0]
#         new_inputs_embeds_padded.append(
#             torch.cat((cur_new_embed,
#                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
#                                    dtype=cur_new_embed.dtype,
#                                    device=cur_new_embed.device)),
#                       dim=0))
#         if cur_len > 0:
#             new_labels_padded[i, :cur_len] = cur_new_labels
#             attention_mask[i, :cur_len] = True
#             position_ids[i, :cur_len] = torch.arange(
#                 0,
#                 cur_len,
#                 dtype=position_ids.dtype,
#                 device=position_ids.device)

#     new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0)

#     if _labels is None:
#         new_labels = None
#     else:
#         new_labels = new_labels_padded

#     if _attention_mask is None:
#         attention_mask = None
#     else:
#         attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

#     if _position_ids is None:
#         position_ids = None

#     return {
#         'input_ids': None,
#         'position_ids': position_ids,
#         'attention_mask': attention_mask,
#         'past_key_values': past_key_values,
#         'inputs_embeds': new_inputs_embeds,
#         'labels': new_labels
#     }



def contains_chinese_characters(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')  # 中文字符的 Unicode 范围
    return bool(pattern.search(text))


def compare_images(image1, image2):
    if np.array_equal(image1, image2):
        return True  # 图片相同
    else:
        return False  # 图片不同

from typing import List, Tuple, Optional, Union
import math
import torch.nn.functional as F

def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    if not torch.jit.is_scripting() and verbose:
        _logger.info(f'Resized position embedding: {old_size} to {new_size}.')

    return posemb


def interpolate_pos_embed(model, new_image_size):
    pos_embed =  model.vision_model.embeddings.position_embedding.weight

    embedding_size = pos_embed.shape[-1]
    num_patches = int(model.vision_model.embeddings.image_size / model.vision_model.embeddings.patch_size)
    num_extra_tokens = pos_embed.shape[-2] - (num_patches ** 2)
    orig_size = num_patches
    new_size = int(new_image_size / model.vision_model.embeddings.patch_size)

    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed[:num_extra_tokens, :]
        # only the position tokens are interpolated
        pos_tokens = pos_embed[num_extra_tokens:, :]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens.float(), size=(new_size, new_size), mode='bicubic', align_corners=False).to(dtype=extra_tokens.dtype)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze()
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)

    return new_pos_embed

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument(
        '--model_name_or_path', default='/mnt/pfs-guan-ssai/cv/sunhaoyi/dolphin-2.9.1-yi-1.5-34b', help='Hugging Face model name or path')
    adapter_group = parser.add_mutually_exclusive_group()
    adapter_group.add_argument(
        '--adapter', default=None, help='adapter name or path')
    adapter_group.add_argument(
        '--llava', default='/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/dolphinyi34b-siglipdino-20m-ft_epoch_1_hf', help='llava name or path')
    parser.add_argument(
        '--visual-encoder', default='/mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384', help='visual encoder name or path')
    parser.add_argument(
        '--dino-visual-encoder', default='/mnt/pfs-guan-ssai/cv/cjy/models/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c', help='visual encoder name or path')
    parser.add_argument(
        '--visual-select-layer', default=-2, help='visual select layer')
    parser.add_argument('--image', default="view.jpg", help='image')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default='qwen_chat',
        help='Specify a prompt template')
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument(
        '--system', default=None, help='Specify the system text')
    system_group.add_argument(
        '--system-template',
        choices=SYSTEM_TEMPLATE.keys(),
        default=None,
        help='Specify a system template')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--with-plugins',
        nargs='+',
        choices=['calculate', 'solve', 'search'],
        help='Specify plugins to use')
    parser.add_argument(
        '--no-streamer', action='store_true', help='Whether to with streamer')
    parser.add_argument(
        '--lagent', action='store_true', help='Whether to use lagent')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
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
        '--temperature',
        type=float,
        default=0.1,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
        'keep for top-k-filtering.')
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.75,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    torch.manual_seed(args.seed)

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

    if args.with_plugins is None:
        inner_thoughts_open = False
        calculate_open = False
        solve_open = False
        search_open = False
    else:
        assert args.prompt_template == args.system_template == 'moss_sft'
        from plugins import plugins_api
        inner_thoughts_open = True
        calculate_open = 'calculate' in args.with_plugins
        solve_open = 'solve' in args.with_plugins
        search_open = 'search' in args.with_plugins
        # pre-import for api and model preparation
        if calculate_open:
            from plugins import calculate  # noqa: F401
        if solve_open:
            from plugins import solve  # noqa: F401
        if search_open:
            from plugins import search  # noqa: F401
    # build llm
    llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, 
                                               **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        encode_special_tokens=True
    )
    print(f'Load LLM from {args.model_name_or_path}')
    if args.adapter is not None:
        llm = PeftModel.from_pretrained(
            llm,
            args.adapter,
            offload_folder=args.offload_folder,
            trust_remote_code=True)
        print(f'Load adapter from {args.adapter}')
    if args.llava is not None:
        llava_path = snapshot_download(
            repo_id=args.llava) if not osp.isdir(
                args.llava) else args.llava
        
        # build visual_encoder
        if 'visual_encoder' in os.listdir(llava_path):
            assert args.visual_encoder is None, (
                "Please don't specify the `--visual-encoder` since passed "
                '`--llava` contains a visual encoder!')
            visual_encoder_path = osp.join(llava_path, 'visual_encoder')
        else:
            assert args.visual_encoder is not None, (
                'Please specify the `--visual-encoder`!')
            visual_encoder_path = args.visual_encoder

        if args.dino_visual_encoder is not None:
            dino_visual_encoder_path = args.dino_visual_encoder
        else:
            dino_visual_encoder_path = None

        # visual_encoder = CLIPVisionModel.from_pretrained(
        #     visual_encoder_path,
        #     torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype])
        # image_processor = CLIPImageProcessor.from_pretrained(
        #     visual_encoder_path)

        dynamic_image_size = 448

        if 'EVA'  in visual_encoder_path:
            print("use eva clip")
    
            if dynamic_image_size is not None:
                hf_config = AutoConfig.from_pretrained(visual_encoder_path, trust_remote_code=True)
                hf_config.dynamic_image_size = dynamic_image_size

                print(f'interpolate pos embed for eva to {dynamic_image_size}')

                visual_encoder = AutoModel.from_pretrained(visual_encoder_path, config=hf_config, torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype], device_map='cpu', trust_remote_code=True, 
                                                           ignore_mismatched_sizes=True)

                position_embedding_weight = resample_abs_pos_embed(
                    visual_encoder.vision_model.embeddings.position_embedding.weight.unsqueeze(0),
                    (dynamic_image_size // visual_encoder.vision_model.embeddings.patch_size, dynamic_image_size // visual_encoder.vision_model.embeddings.patch_size),
                    num_prefix_tokens=visual_encoder.vision_model.embeddings.num_prefix_tokens,
                ).squeeze(0)

            else:
                hf_config = AutoConfig.from_pretrained(visual_encoder_path, trust_remote_code=True)
                visual_encoder = AutoModel.from_pretrained(visual_encoder_path, config=hf_config, torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype], device_map='cpu', trust_remote_code=True, 
                                                        ignore_mismatched_sizes=True)

        elif 'siglip' in visual_encoder_path:

            print("use siglip clip")
            visual_encoder = SiglipVisionModel.from_pretrained(visual_encoder_path, torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype], device_map='cpu')
            
            if dynamic_image_size is not None:
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

        else: 
            print("use open clip")
            visual_encoder = CLIPVisionModel.from_pretrained(visual_encoder_path, torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype], device_map='cpu')

            if dynamic_image_size is not None:
                print(f'interpolate pos embed for clip to {dynamic_image_size}')
                position_embedding_weight = interpolate_pos_embed(visual_encoder, dynamic_image_size)

        if dynamic_image_size is not None:
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
                torch_dtype=torch.float16,
                device_map='cpu',
                trust_remote_code=True, 
                # ignore_mismatched_sizes=True
            )
            print(f'Load dino_visual_encoder from {dino_visual_encoder_path}')
        else:
            dino_visual_encoder = None

        dino_image_processor = None
        if dynamic_image_size is not None:
            image_processor = CLIPImageProcessor.from_pretrained(visual_encoder_path, crop_size=dynamic_image_size, size=dynamic_image_size)
            # if dino_visual_encoder_path is not None:
            #     dino_image_processor = CLIPImageProcessor.from_pretrained(dino_visual_encoder_path, crop_size=dynamic_image_size, size=dynamic_image_size)
        elif visual_encoder.__class__.__name__ == 'SiglipVisionModel':
            image_processor = CLIPImageProcessor.from_pretrained(visual_encoder_path, crop_size=384, size=384)
            # if dino_visual_encoder_path is not None:
            #     dino_image_processor = CLIPImageProcessor.from_pretrained(dino_visual_encoder_path, crop_size=384, size=384)
        else:
            image_processor = CLIPImageProcessor.from_pretrained(visual_encoder_path)

        print(f'image_processor: {image_processor}')
        print(f"dino_image_processor: {dino_image_processor}")

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
                offload_folder=args.offload_folder)
            print(f'Load visual_encoder adapter from {adapter_path}')
        if 'dino_visual_encoder_adapter' in os.listdir(llava_path):
            adapter_path = osp.join(llava_path, 'dino_visual_encoder_adapter')
            dino_visual_encoder = PeftModel.from_pretrained(dino_visual_encoder,
                                                            adapter_path,
                                                            offload_folder=args.offload_folder,
                                                            trust_remote_code=True,
                                                            # device_map='cpu'
                                                            )
            print(f'Load dino_visual_encoder_adapter adapter from {adapter_path}')

        # build projector
        projector_path = osp.join(llava_path, 'projector')
        projector = AutoModel.from_pretrained(
            projector_path,
            torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],
            trust_remote_code=True)
        print(f'Load projector from {projector_path}')

        projector.cuda()
        projector.eval()
        visual_encoder.cuda()
        visual_encoder.eval()

        if dino_visual_encoder:
            dino_visual_encoder.cuda()
            dino_visual_encoder.eval()

    llm.eval()

    if args.image is not None:
        print('Multimode Conversation')

    num_sub_images = 6
    if num_sub_images > 0:
            print(f"inference with multi subimages")

    stop_words = args.stop_words
    sep = ''
    if args.prompt_template:
        template = PROMPT_TEMPLATE[args.prompt_template]
        stop_words += template.get('STOP_WORDS', [])
        sep = template.get('SEP', '')
    stop_criteria = get_stop_criteria(
        tokenizer=tokenizer, stop_words=stop_words)

    if args.no_streamer:
        Streamer = None
    else:
        Streamer = get_streamer(llm)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    def inference(image, text):
        global last_turn_image_input
        global n_turn
        global inputs
        global format_inputs

        if last_turn_image_input is None:
            last_turn_image_input = image
        
        same_turn = compare_images(image, last_turn_image_input)
        # print(f"same_turn: {same_turn}")

        if not same_turn:
            last_turn_image_input = image
            n_turn = 0
            inputs = ''
            format_inputs = ''

        # print(f"n_turn: {n_turn}")
        # print(f"inputs: {[inputs]}")

        format_inputs += f"第{n_turn + 1}轮对话：\n"

        # if contains_chinese_characters(text):
        #     text = f"用中文回答下面问题：{text}"

        print(image, text)

        if image is not None and n_turn == 0:
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

        if args.prompt_template:
            prompt_text = ''
            template = PROMPT_TEMPLATE[args.prompt_template]
            if 'SYSTEM' in template and n_turn == 0:
                system_text = None
                if args.system_template is not None:
                    system_text = SYSTEM_TEMPLATE[
                        args.system_template].format(
                            round=n_turn + 1, bot_name=args.bot_name)
                elif args.system is not None:
                    system_text = args.system
                if system_text is not None:
                    prompt_text += template['SYSTEM'].format(
                        system=system_text,
                        round=n_turn + 1,
                        bot_name=args.bot_name)
            
            prompt_text += template['INSTRUCTION'].format(
                input=text, round=n_turn + 1, bot_name=args.bot_name)
            if template.get('TYPE', '') == 'mindgpt' and n_turn == 0:
                system = template['SYSTEM'].format(system='你是一个名字叫做理想同学的AI数字生命体。')
                prompt_text = system + prompt_text

            if args.prompt_template == args.system_template == 'moss_sft':
                if not inner_thoughts_open:
                    prompt_text.replace('- Inner thoughts: enabled.',
                                        '- Inner thoughts: disabled.')
                if not calculate_open:
                    prompt_text.replace(('- Calculator: enabled. API: '
                                            'Calculate(expression)'),
                                        '- Calculator: disabled.')
                if not solve_open:
                    prompt_text.replace(
                        '- Equation solver: enabled. API: Solve(equation)',
                        '- Equation solver: disabled.')
                if not search_open:
                    prompt_text.replace(
                        '- Web search: enabled. API: Search(query)',
                        '- Web search: disabled.')
        else:
            prompt_text = text

        inputs += prompt_text
        print(f"inputs: {[inputs]}")
        format_inputs += prompt_text.replace("<image>\n", "").replace("<|im_start|>", "").replace("<|im_end|>", "")
        if args.image is None:
            if n_turn == 0:
                ids = tokenizer.encode(inputs, return_tensors='pt')
            else:
                ids = tokenizer.encode(
                    inputs, return_tensors='pt', add_special_tokens=False)
            streamer = Streamer(
                tokenizer) if Streamer is not None else None
            if args.with_plugins is not None:
                generate_output = llm.generate(
                    inputs=ids.cuda(),
                    generation_config=gen_config,
                    streamer=streamer,
                    stopping_criteria=stop_criteria).cpu()
                generate_output_text = tokenizer.decode(
                    generate_output[0][len(ids[0]):])
                if streamer is None:
                    end = '' if generate_output_text[-1] == '\n' else '\n'
                    print(generate_output_text, end=end)
                pattern = r'<\|Commands\|>:(.*?)<eoc>'
                command_text = ', '.join(
                    re.findall(pattern, generate_output_text))
                extent_text = plugins_api(
                    command_text,
                    calculate_open=calculate_open,
                    solve_open=solve_open,
                    search_open=search_open)
                end = '' if extent_text[-1] == '\n' else '\n'
                print(extent_text, end=end)
                extent_text_ids = tokenizer.encode(
                    extent_text,
                    return_tensors='pt',
                    add_special_tokens=False)
                new_ids = torch.cat((generate_output, extent_text_ids),
                                    dim=1)
                new_streamer = Streamer(
                    tokenizer) if Streamer is not None else None
                generate_output = llm.generate(
                    inputs=new_ids.cuda(),
                    generation_config=gen_config,
                    streamer=new_streamer,
                    stopping_criteria=stop_criteria)
                if streamer is None:
                    output_text = tokenizer.decode(
                        generate_output[0][len(new_ids[0]):])
                    end = '' if output_text[-1] == '\n' else '\n'
                    print(output_text, end=end)
            else:           
                generate_output = llm.generate(
                    inputs=ids.cuda(),
                    generation_config=gen_config,
                    streamer=streamer,
                    stopping_criteria=stop_criteria)
                if streamer is None:
                    output_text = tokenizer.decode(
                        generate_output[0][len(ids[0]):])
                    end = '' if output_text[-1] == '\n' else '\n'
                    print(output_text, end=end)
            inputs = tokenizer.decode(generate_output[0])
        else:
            image = load_image(image)
            if num_sub_images > 0:
                try:
                    input_size = image_processor.crop_size['height']
                except Exception as e:
                    input_size = image_processor.crop_size
                images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=num_sub_images)
                num_pixel_values = len(images)
                image = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
                image = image.cuda()

                if dino_image_processor:
                    image_dino = dino_image_processor.preprocess(images, return_tensors='pt')['pixel_values']
                    image_dino = image_dino.cuda()
                else:
                    image_dino = image

            else:
                square_image = expand2square(
                    image, tuple(int(x * 255) for x in image_processor.image_mean))
                image = image_processor.preprocess(
                    image, return_tensors='pt')['pixel_values'][0]
                image = image.cuda().unsqueeze(0)

                if dino_image_processor:
                    image_dino = image_processor.preprocess(
                        square_image, return_tensors='pt')['pixel_values'][0]
                    image_dino = image_dino.cuda().unsqueeze(0)

            print(f"image: {image.shape}")
            print(f"image_dino: {image_dino.shape}")
            print(f"use two image processor: {(image - image_dino).mean() != 0}")

            image = image.to(visual_encoder.dtype)
            visual_outputs = visual_encoder(image, output_hidden_states=True)

            if dino_visual_encoder:
                image_dino = image_dino.to(dino_visual_encoder.dtype)
                dino_outputs = dino_visual_encoder(image_dino, output_hidden_states=True)
            if dino_visual_encoder:
                vit_pixel_values = visual_outputs.hidden_states[args.visual_select_layer]
                dino_pixel_values = dino_outputs.hidden_states[args.visual_select_layer][:, 1:]
                print(f"vit_pixel_values: {vit_pixel_values.shape}, dino_pixel_values: {dino_pixel_values.shape}")
                pixel_values = projector(torch.cat([vit_pixel_values, dino_pixel_values], dim=-1))
            else:
                pixel_values = visual_outputs.hidden_states[args.visual_select_layer][:, 1:].to(projector.dtype)
                pixel_values = projector(pixel_values)
                pixel_values = pixel_values.to(llm.dtype)
    
            print(f"pixel_values: {pixel_values.shape}")
                
            chunk_encode = []
            for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
                if idx == 0 and n_turn == 0:
                    cur_encode = tokenizer.encode(chunk)
                else:
                    cur_encode = tokenizer.encode(
                        chunk, add_special_tokens=False)
                chunk_encode.append(cur_encode)
            assert len(chunk_encode) == 2
            ids = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                ids.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    ids.append(IMAGE_TOKEN_INDEX)

            # ids = tokenizer.encode(inputs)

            # print(f"ids: {ids}")

            ids = torch.tensor(ids).cuda().unsqueeze(0)
            num_ids = ids.shape[-1]
            if num_sub_images> 1:
                print(f"num_pixel_values: {num_pixel_values}")
                pixel_values = pixel_values.view(-1, num_pixel_values, pixel_values.shape[1], pixel_values.shape[2])
                mm_inputs = prepare_multi_subimages_inputs_labels_for_multimodal(
                    llm=llm,
                    input_ids=ids,
                    pixel_values=pixel_values, 
                    num_pixel_values=[num_pixel_values])
            else:
                mm_inputs = prepare_inputs_labels_for_multimodal(
                    llm=llm, input_ids=ids, pixel_values=pixel_values)

            streamer = Streamer(
                tokenizer) if Streamer is not None else None
            
            generate_output = llm.generate(
                **mm_inputs,
                generation_config=gen_config,
                streamer=streamer,
                bos_token_id=tokenizer.bos_token_id,
                stopping_criteria=stop_criteria)
            
            if streamer is None:
                output_text = tokenizer.decode(generate_output[0])
                end = '' if output_text[-1] == '\n' else '\n'
                # print(output_text, end=end)
            # print(generate_output[0])
            # print(tokenizer.decode(generate_output[0]))
            num_generate_output = generate_output[0].shape[0]
            inputs += tokenizer.decode(generate_output[0])
            format_inputs += tokenizer.decode(generate_output[0]).replace("<|im_start|>", "").replace("<|im_end|>", "")
        n_turn += 1
        inputs += sep
        format_inputs += sep

        if len(generate_output[0]) >= args.max_new_tokens:
            print(
                'Remove the memory of history responses, since '
                f'it exceeds the length limitation {args.max_new_tokens}.')
            n_turn = 0
            inputs = ''
            format_inputs = ''

        format_inputs = format_inputs.replace("user\n", "用户：").replace("assistant\n", "mindGPT-VL：")
        format_inputs += f"{'='*50}\n"

        return format_inputs

        
    with gr.Blocks() as demo:
        # gr.Markdown("""\
        # <p align="center"><img src="https://modelscope.cn/api/v1/models/qwen/Qwen-7B-Chat/repo?
        # Revision=master&FilePath=assets/logo.jpeg&View=true" style="height: 80px"/><p>""")
        gr.Markdown("""<center><font size=8>mindGPT-VL-Chat Bot</center>""")
        gr.Markdown(
                    """\
        <center><font size=3>This WebUI is based on mindGPT-VL-Chat, developed by Li Xiang. \
        (本WebUI基于mindGPT-VL-Chat打造，实现聊天机器人功能。)</center>""")
        # gr.Markdown("""\
        # <center><font size=4>Qwen-VL <a href="https://modelscope.cn/models/qwen/Qwen-VL/summary">?? </a> 
        # | <a href="https://huggingface.co/Qwen/Qwen-VL">??</a>&nbsp ｜ 
        # Qwen-VL-Chat <a href="https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary">?? </a> | 
        # <a href="https://huggingface.co/Qwen/Qwen-VL-Chat">??</a>&nbsp ｜ 
        # &nbsp<a href="GitHub - QwenLM/Qwen-VL: The official repo of Qwen-VL (通义千问-VL) chat & pretrained large vision langu">Github</a></center>""")

        prompt_textbox = gr.Textbox(label="Prompt:", placeholder="prompt", lines=2)
        image_input = gr.Image(type="filepath")

        print(type(image_input))

        interface=gr.Interface(
            fn=inference,
            inputs=[image_input, prompt_textbox],
            outputs="text",
            allow_flagging="never",
        )
    demo.launch(server_name='0.0.0.0', server_port=6006)


if __name__ == '__main__':
    main()



