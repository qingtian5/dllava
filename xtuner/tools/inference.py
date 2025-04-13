import os
import os.path as osp
import string
import sys
import warnings
import cv2
import numpy as np

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer,
                        #   CLIPImageProcessor, 
                          CLIPVisionModel, SiglipVisionModel,
                          GenerationConfig, StoppingCriteriaList)

from xtuner.utils import CV_CLIPImageProcessor as CLIPImageProcessor
from xtuner.model.utils import prepare_multi_subimages_inputs_labels_for_multimodal
from xtuner.dataset.llava_multisubimages import dynamic_preprocess
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)
from xtuner.tools.utils import get_stop_criteria, get_streamer


TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def main():
    model_name_or_path = '/mnt/pfs-guan-ssai/cv/sunhaoyi/dolphin-2.9.1-yi-1.5-34b-unused'
    visual_encoder='/mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384'
    dino_visual_encoder='/mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large'
    llava='/mnt/pfs-mc0p4k/cv/team/jucic/xtuner/work_dirs/llava_dolphin_yi_34b_siglipdino_lora_e1_gpu8_finetune_multisubimages/dolphinyi34b-siglipdino-123m-ft-0820_all_sft_1043k_338kgen_100knlu/epoch_1_hf'
    prompt_template="qwen_chat"
    
    # model_name_or_path = '/mnt/pfs-guan-ssai/cv/sunhaoyi/MindGPT2.0-32K'
    # visual_encoder='/mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384'
    # dino_visual_encoder='/mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large'
    # llava='/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/work_dirs/llava_mindgpt_moe_siglipdino_lora_e1_gpu8_finetune_multisubimages/mindgpt-moe-2.0-32k-siglipdino-558k-ft_epoch_1_hf'
    # prompt_template="mindgpt"

    dynamic_image_size=448 
    num_sub_images=6
    torch_dtype='fp16'
    visual_select_layer=-2


    # build llm
    quantization_config = None
    load_in_8bit = False
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': 'auto',
        'offload_folder': None,
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[torch_dtype]
    }

    # build llm
    llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                               **model_kwargs)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        encode_special_tokens=True)
    print(f'Load LLM from {model_name_or_path}')

    llava_path = snapshot_download(
        repo_id=llava) if not osp.isdir(llava) else llava
    
    # visual_encoder = model.visual_encoder

    # build visual_encoder
    if 'visual_encoder' in os.listdir(llava_path):
        assert visual_encoder is None, (
            "Please don't specify the `--visual-encoder` since passed "
            '`--llava` contains a visual encoder!')
        visual_encoder_path = osp.join(llava_path, 'visual_encoder')
    else:
        assert visual_encoder is not None, (
            'Please specify the `--visual-encoder`!')
        visual_encoder_path = visual_encoder

    if dino_visual_encoder is not None:
        dino_visual_encoder_path = dino_visual_encoder
    else:
        dino_visual_encoder_path = None


    dynamic_image_size = dynamic_image_size

    if 'siglip' in visual_encoder_path:

        print("use siglip clip")
        visual_encoder = SiglipVisionModel.from_pretrained(visual_encoder_path, 
                                                            torch_dtype=torch.float16, 
                                                            device_map='cpu')

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

    print(f'image_processor: {image_processor}')
    print(f"dino_image_processor: {dino_image_processor}")

    # load adapter
    if 'llm_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'llm_adapter')
        llm = PeftModel.from_pretrained(
            llm,
            adapter_path,
            offload_folder=None,
            trust_remote_code=True)
        print(f'Load LLM adapter from {adapter_path}')
    if 'visual_encoder_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'visual_encoder_adapter')
        visual_encoder = PeftModel.from_pretrained(
            visual_encoder,
            adapter_path,
            offload_folder=None,
            trust_remote_code=True)
        print(f'Load visual_encoder adapter from {adapter_path}')

    if 'dino_visual_encoder_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'dino_visual_encoder_adapter')
        dino_visual_encoder = PeftModel.from_pretrained(dino_visual_encoder,
                                                        adapter_path,
                                                        offload_folder=None,
                                                        trust_remote_code=True,
                                                        # device_map='cpu'
                                                        )
        print(f'Load dino_visual_encoder_adapter adapter from {adapter_path}')

    # build projector
    projector_path = osp.join(llava_path, 'projector')
    print(f"projector_path: {projector_path}")
    projector = AutoModel.from_pretrained(
        projector_path,
        torch_dtype=TORCH_DTYPE_MAP[torch_dtype],
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

    num_sub_images = num_sub_images
    if num_sub_images > 0:
            print(f"inference with multi subimages")

    stop_words = []
    if prompt_template:
        template = PROMPT_TEMPLATE[prompt_template]
        stop_words += template.get('STOP_WORDS', [])
    stop_criteria = get_stop_criteria(
        tokenizer=tokenizer, stop_words=stop_words)

    gen_config = GenerationConfig(
        max_new_tokens=1024,
        temperature=0.7,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
        # repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        
    )

    print(f"gen_config: {gen_config}")

    # text = '帮我看看图片里面有什么东西？'
    text = '帮我看看这个用英语怎么说？'
    text = DEFAULT_IMAGE_TOKEN + '\n' + text



    if prompt_template:
        prompt_text = ''
        template = PROMPT_TEMPLATE[prompt_template]
        prompt_text += template['INSTRUCTION'].format(
            input=text, round=1, bot_name='BOT')
    else:
        prompt_text = text

    system = template['SYSTEM'].format(
        system='你是一个名字叫做理想同学的AI数字生命体。\n作为一款多模态AI，你有处理和分析图像的能力。你必须非常谨慎的审查对话中的图像，并在构建答案时考虑问题内容。\n对于非常简单的问题，你应该提供简洁的回答，但对于复杂和开放式的问题，则应提供详尽准确的回答')

    prompt_text = system + prompt_text

    inputs = prompt_text

    # history = '\n<|im_start|>user\n帮我看看图片里有什么东西<|im_end|>\n<|im_start|>assistant\n图片中的物体是一个电源供应器，通常用于为电子设备提供稳定的电源。电源供应器通常具有调节电压和电流的功能，以确保设备的安全运行。这个电源供应器有一个数字显示屏，用于显示电压和电流值。在显示屏上方，有红色和黑色的旋钮，用于调节电压和电流。\n电源供应器通常用于实验室、电子维修、DIY项目或任何需要精确电源控制的环境。它们可以提供直流电（DC）或交流电（AC），具体取决于其设计。电源供应器是电子工程师和爱好者必备的工具，因为它允许他们测试和验证电路，而不必依赖电池或不可靠的电源。\n在背景中，我们可以看到一个实验室或工作台，这进一步支持了电源供应器用于电子实验或维修的假设。<|im_end|>\n\n'
    # inputs = history + inputs
    # print(f"inputs: {[inputs]}")

    
    image_path = '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/soda.jpeg'
    image = cv2.imread(image_path).astype(np.uint8)

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

    print(f"image: {image.shape}")
    print(f"image_dino: {image_dino.shape}")
    print(f"use two image processor: {(image - image_dino).mean() != 0}")

    image = image.to(visual_encoder.dtype)
    visual_outputs = visual_encoder(image, output_hidden_states=True)
    if dino_visual_encoder:
        image_dino = image_dino.to(dino_visual_encoder.dtype)
        dino_outputs = dino_visual_encoder(image_dino, output_hidden_states=True)

    if dino_visual_encoder:
        vit_pixel_values = visual_outputs.hidden_states[visual_select_layer]
        dino_pixel_values = dino_outputs.hidden_states[visual_select_layer][:, 1:]
        print(f"vit_pixel_values: {vit_pixel_values.shape}, dino_pixel_values: {dino_pixel_values.shape}")
        pixel_values = projector(torch.cat([vit_pixel_values, dino_pixel_values], dim=-1))

    print(f"pixel_values: {pixel_values.shape}")

    chunk_encode = []
    for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
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
    print(f"ids: {ids}")

    if num_sub_images> 1:
        print(f"num_pixel_values: {num_pixel_values}")
        pixel_values = pixel_values.view(-1, num_pixel_values, pixel_values.shape[1], pixel_values.shape[2])
        mm_inputs = prepare_multi_subimages_inputs_labels_for_multimodal(
            llm=llm,
            input_ids=ids,
            pixel_values=pixel_values, 
            num_pixel_values=[num_pixel_values])

    Streamer = get_streamer(llm) 
    streamer = Streamer(
        tokenizer) if Streamer is not None else None
    
    generate_output = llm.generate(
        **mm_inputs,
        generation_config=gen_config,
        streamer=streamer,
        bos_token_id=tokenizer.bos_token_id,
        stopping_criteria=stop_criteria)
    
    # print(f"generate_output[0]: {generate_output[0]}")

    predict = tokenizer.decode(
        generate_output[0], skip_special_tokens=True).strip()
    # print(f"predict: {[predict]}")         

    print(f'\n{"="*50}\nSample output:\n'
          f'{inputs + predict}\n{"="*50}\n')       


if __name__ == '__main__':
    main()
