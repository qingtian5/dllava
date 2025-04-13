# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, 
                        #   CLIPImageProcessor, CLIPVisionModel, 
                          SiglipImageProcessor, SiglipVisionModel, 
                          AutoModel, AutoConfig, AutoProcessor)

from xtuner.utils import CV_CLIPImageProcessor as CLIPImageProcessor

from xtuner.dataset import LLaVADataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine import DatasetInfoHook, EvaluateChatHook
from xtuner.model import LLaVAModel
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
# llm_name_or_path = '/mnt/pfs-guan-ssai/cv/moxu/mindgpt-v6/'
llm_name_or_path='/mnt/pfs-guan-ssai/cv/moxu/MindGPT-V6-16B-stage2-26000/'
# visual_encoder_name_or_path = '/mnt/pfs-guan-ssai/cv/cjy/models/mindvit/2024_04_03/EVA02-l-14-224-20240403'
visual_encoder_name_or_path = '/mnt/pfs-guan-ssai/cv/cjy/models/mindvit/2024_04_14/EVA02-l-14-336-20240414'
# visual_encoder_name_or_path='/mnt/pfs-guan-ssai/cv/cjy/models/mindvit/2024_04_15/MindViT_CLIP_L_336_20240415'

# Specify the pretrained pth
# pretrained_pth = './work_dirs/llava_mindgptv6_base_16b_qlora_eva_vit_large_p14_336_lora_e1_gpu8_mixpretrain/mindgptv6-26000-eva0414-vicuna-allava-mix17m/epoch_1.pth'
pretrained_pth = "./work_dirs/llava_mindgptv6_base_16b_qlora_eva_vit_large_p14_336_lora_e1_gpu8_finetune/VL1.2m_hy_allava_bunny/epoch_1.pth/"


# Data
data_root = './data/llava_data/'
# data_path = data_root + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
# image_folder = data_root + 'llava_images'
# data_path = data_root + 'LLaVA-Pretrain/toy.json'
data_path = "/mnt/pfs-guan-ssai/cv/moxu/xtuner-main/data/llava_data/LLaVA-Instruct-150K/incar0401_pro_5.json"
image_folder = ''
prompt_template = PROMPT_TEMPLATE.vicuna
max_length = int(2048 - (448 / 14)**2)


# work_dir = './work_dirs/llava_mindgptv6_base_16b_qlora_eva_vit_large_p14_336_lora_e1_gpu8_finetune/mindgptv6-26000-eva0414-vicuna-allava-mix17m-ft'
work_dir = './work_dirs/llava_mindgptv6_base_16b_qlora_eva_vit_large_p14_336_lora_e1_gpu8_finetune/incar4-VL1.2m_hy_allava_bunny'

# Scheduler & Optimizer
batch_size = 8  # per_device
accumulative_counts = 1
dataloader_num_workers = 0
max_epochs = 10
optim_type = AdamW
lr = 1e-6
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Evaluate the generation performance during the training
evaluation_freq = 100
SYSTEM = ''
evaluation_images = 'view.jpg'
evaluation_inputs = ['请描述一下这张照片', 'Please describe this picture']

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
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
    size=448
    )

model = dict(
    type=LLaVAModel,
    freeze_llm=True,
    freeze_visual_encoder=True,
    pretrained_pth=pretrained_pth,
    dynamic_image_size=448,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),

    llm_lora=dict(
        type=LoraConfig,
        r=512,
        lora_alpha=256,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'),

    visual_encoder=dict(
        type=AutoModel.from_pretrained,
        config=dict(
            type=AutoConfig.from_pretrained,
            pretrained_model_name_or_path=visual_encoder_name_or_path,
            trust_remote_code=True),
        pretrained_model_name_or_path=visual_encoder_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        ignore_mismatched_sizes=True
    ),

    visual_encoder_lora=dict(
        type=LoraConfig, r=64, lora_alpha=16, lora_dropout=0.05, bias='none')
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
llava_dataset = dict(
    type=LLaVADataset,
    data_path=data_path,
    image_folder=image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=llava_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=default_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        T_max=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        image_processor=image_processor,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        system=SYSTEM,
        prompt_template=prompt_template)
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 100 iterations.
    logger=dict(type=LoggerHook, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per epoch.
    checkpoint=dict(type=CheckpointHook, interval=1),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)