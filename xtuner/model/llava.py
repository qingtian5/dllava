# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import distributed as dist
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine import print_log
from peft import get_peft_model, prepare_model_for_kbit_training

from xtuner.registry import BUILDER
from .modules import dispatch_modules
from .modules import (ProjectorConfig, ProjectorModel,
                      DualPathProjectorConfig, DualPathProjectorModel,
                      FusionProjectorConfig, FusionProjectorModel,
                      HoneybeeProjectorConfig, HoneybeeProjectorModel, 
                      LDPProjectorConfig, LDPProjectorModel, 
                      UHDProjectorConfig, UHDProjectorModel, 
                      Idefics2ProjectorConfig, Idefics2ProjectorModel, 
                      TokenPackerConfig, TokenPackerProjectorModel)
from .utils import (LoadWoInit, find_all_linear_names, find_all_linear_conv_names, find_lm_head_names, find_all_linear_names_with_lm_head,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, prepare_multi_subimages_inputs_labels_for_multimodal,
                    traverse_dict)

from transformers import AutoModel, AutoConfig, SiglipVisionModel, ConvNextConfig, ConvNextModel
import open_clip
from safetensors import safe_open
from einops import rearrange
import timm

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
        print_log("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size), 'current')
        extra_tokens = pos_embed[:num_extra_tokens, :]
        # only the position tokens are interpolated
        pos_tokens = pos_embed[num_extra_tokens:, :]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)

        orig_dtype = pos_tokens.dtype
        pos_tokens = pos_tokens.float()
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.to(orig_dtype) 
    
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze()
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)

    return new_pos_embed


class LLaVAModel(BaseModel):

    def __init__(self,
                 llm,
                 visual_encoder,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 freeze_visualenc_lora=False,
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_type='mlp',
                 projector_depth=2,
                 llm_lora=None,
                 llm_lora_adapt_mode=False,
                 visual_encoder_lora=None,
                 activate_convnext_lora=True,
                 dino_visual_encoder=None,
                 convnext_visual_encoder=None,
                 use_activation_checkpointing=True,
                 dynamic_image_size=None,
                 num_sub_images=0,
                 add_img_slice_special_tokens=False,
                 fusion_channel=False,
                 mark_slice_indices=False
                #  dense_connect=False
                 ):
        super().__init__()
        # print("====> build LLaVAModel")
        # print(f"====> llm: {llm}")

        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_visualenc_lora = freeze_visualenc_lora
        self.activate_convnext_lora = activate_convnext_lora
        self.visual_select_layer = visual_select_layer
        self.num_sub_images = num_sub_images
        self.add_img_slice_special_tokens = add_img_slice_special_tokens
        self.fusion_channel = fusion_channel
        self.mark_slice_indices = mark_slice_indices

        # self.dense_connect = dense_connect

        with LoadWoInit():
            self.llm = self._build_from_cfg_or_module(llm)
            self.llm_name = self.llm.__class__.__name__
            print_log(f"build llm {self.llm_name} ", 'current')

            # self.visual_encoder = self._build_from_cfg_or_module(visual_encoder)

            if dynamic_image_size is None:
                self.visual_encoder = self._build_from_cfg_or_module(visual_encoder)
            else:

                vit_type = 'eva-vit' if 'cjy' in visual_encoder['pretrained_model_name_or_path'] else 'clip-vit'

                if vit_type == 'clip-vit':
                    
                    # print_log(f'build clip vit model', 'current')
                    
                    # visual_encoder_configuration = AutoConfig.from_pretrained(visual_encoder['pretrained_model_name_or_path']).vision_config
                    # visual_encoder_configuration.image_size = dynamic_image_size
                    # self.visual_encoder = visual_encoder['type'](config=visual_encoder_configuration).to(dtype=visual_encoder['torch_dtype']) 

                    # print(f"after init: {self.visual_encoder.vision_model.embeddings.position_embedding.weight.shape}, {self.visual_encoder.vision_model.embeddings.position_embedding.weight.mean()}")

                    # vit_model_state_dict = {}

                    # with safe_open(f"{visual_encoder['pretrained_model_name_or_path']}/model.safetensors", framework="pt", device="cpu") as f:
                    #     for k in f.keys():
                    #         if 'vision_model' in k:
                    #             vit_model_state_dict[k] = f.get_tensor(k)


                    # pos_embed = vit_model_state_dict['vision_model.embeddings.position_embedding.weight']
                    # pos_embed = pos_embed.float()
                    # embedding_size = pos_embed.shape[-1]
                    # orig_size = int(pos_embed.shape[0] ** 0.5)
                    # new_size = int(dynamic_image_size / self.visual_encoder.vision_model.embeddings.patch_size)

                    # print_log("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size), 'current')

                    # pos_embed = pos_embed.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    # pos_embed = torch.nn.functional.interpolate(
                    #     pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    # position_embedding_weight = pos_embed.permute(0, 2, 3, 1).flatten(1, 2).squeeze()

                    # print(f"position_embedding_weight: {position_embedding_weight.shape}, {position_embedding_weight.mean()}")

                    # vit_model_state_dict['vision_model.embeddings.position_embedding.weight'] = position_embedding_weight

                    # self.visual_encoder.load_state_dict(vit_model_state_dict)

                    # print(f"after interpolate: {self.visual_encoder.vision_model.embeddings.position_embedding.weight.shape}, {self.visual_encoder.vision_model.embeddings.position_embedding.weight.mean()}")

                    
                    print_log(f'build clip vit model', 'current')

                    # self.visual_encoder = self._build_from_cfg_or_module(visual_encoder)

                    # self.visual_encoder = SiglipVisionModel(
                    #     pretrained_model_name_or_path=visual_encoder['pretrained_model_name_or_path'],
                    #     trust_remote_code=visual_encoder['trust_remote_code'],
                    #     torch_dtype=visual_encoder['torch_dtype'],
                    # )


                    visual_encoder_configuration = AutoConfig.from_pretrained(visual_encoder['pretrained_model_name_or_path']).vision_config
                    self.visual_encoder = SiglipVisionModel(config=visual_encoder_configuration).to(dtype=visual_encoder['torch_dtype']) 

                    vit_model_state_dict = {}
                    with safe_open(f"{visual_encoder['pretrained_model_name_or_path']}/model.safetensors", framework="pt", device="cpu") as f:
                        for k in f.keys():
                            if 'vision_model' in k: 
                                vit_model_state_dict[k] = f.get_tensor(k)

                    self.visual_encoder.load_state_dict(vit_model_state_dict)

                    print_log(f'interpolate pos embed for clip vit model to {dynamic_image_size}', 'current')

                    if self.visual_encoder.__class__.__name__ == 'SiglipVisionModel':
                        pos_embed = self.visual_encoder.vision_model.embeddings.position_embedding.weight
                        pos_embed = pos_embed.float()
                        embedding_size = pos_embed.shape[-1]
                        orig_size = int(self.visual_encoder.vision_model.embeddings.image_size / self.visual_encoder.vision_model.embeddings.patch_size)
                        new_size = int(dynamic_image_size / self.visual_encoder.vision_model.embeddings.patch_size)
                        
                        print_log("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size), 'current')

                        pos_embed = pos_embed.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                        pos_embed = torch.nn.functional.interpolate(
                            pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
                        position_embedding_weight = pos_embed.permute(0, 2, 3, 1).flatten(1, 2).squeeze()
                        position_embedding_weight = position_embedding_weight.to(self.visual_encoder.dtype)

                    else:
                        position_embedding_weight = interpolate_pos_embed(self.visual_encoder, dynamic_image_size)
                
                num_patches = (dynamic_image_size // self.visual_encoder.vision_model.embeddings.patch_size) ** 2

                if self.visual_encoder.__class__.__name__ == 'SiglipVisionModel':
                    print_log(f'replace pos embed for SiglipVisionModel', 'current')
                    num_positions = num_patches
                else:
                    num_positions = num_patches + 1
                position_embedding = torch.nn.Embedding(num_positions, self.visual_encoder.vision_model.embeddings.embed_dim)
                self.visual_encoder.vision_model.embeddings.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent = False)
                position_embedding.weight = torch.nn.Parameter(position_embedding_weight)
                self.visual_encoder.vision_model.embeddings.position_embedding = position_embedding

            if dino_visual_encoder is not None:
                self.dino_visual_encoder = self._build_from_cfg_or_module(dino_visual_encoder)
            else:
                 self.dino_visual_encoder = None

            if convnext_visual_encoder is not None:
                # self.convnext_visual_encoder = open_clip.create_model(convnext_visual_encoder[0], pretrained=convnext_visual_encoder[1]).visual.trunk
                # self.convnext_visual_encoder = self._build_from_cfg_or_module(convnext_visual_encoder)
                self.convnext_visual_encoder = timm.create_model(
                        model_name='convnext_xxlarge.clip_laion2b_soup_ft_in1k',
                        pretrained=True,
                        # features_only=True,
                        pretrained_cfg_overlay=dict(file=convnext_visual_encoder))
            else:
                 self.convnext_visual_encoder = None

            self.visual_encoder_name = self.visual_encoder.__class__.__name__
            self.dino_visual_encoder_name = self.dino_visual_encoder.__class__.__name__ if self.dino_visual_encoder else None
            self.convnext_visual_encoder_name = self.convnext_visual_encoder.__class__.__name__ if self.convnext_visual_encoder else None
            print_log(f"build visual_encoder {self.visual_encoder_name} ", 'current')
            print_log(f"build dino_visual_encoder: {self.dino_visual_encoder_name} ", 'current')
            print_log(f"build convnext_visual_encoder: {self.convnext_visual_encoder_name} ", 'current')

        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        self.projector_type = projector_type

        # # ====> HonybeeProjectorModel
        if projector_type == 'honeybee':
            cfg_pth = '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/honeybee_configs/Honeybee-C-7B-M144-config.json'

            if self.visual_encoder_name == "RADIOModel":
                visual_hidden_size = self.visual_encoder.config.args['teachers'][-1]['input_size']

                if dynamic_image_size is not None:
                    vit_num_tokens = (dynamic_image_size // self.visual_encoder.patch_size) ** 2
                else:
                    vit_num_tokens = (self.visual_encoder.preferred_resolution[0] // self.visual_encoder.patch_size) ** 2
            else:
                visual_hidden_size = self.visual_encoder.config.hidden_size

                if dynamic_image_size is not None:
                    vit_num_tokens = (dynamic_image_size // self.visual_encoder.config.patch_size) ** 2
                else:
                    vit_num_tokens = (self.visual_encoder.config.image_size // self.visual_encoder.config.patch_size) ** 2

            if self.dino_visual_encoder:
                visual_hidden_size += self.dino_visual_encoder.config.hidden_size
            if self.convnext_visual_encoder:
                # visual_hidden_size += self.convnext_visual_encoder.feature_info[-1]['num_chs']
                visual_hidden_size += self.convnext_visual_encoder.config.hidden_sizes[-1]

            projector_config = HoneybeeProjectorConfig(
                config_path=cfg_pth,
                vit_num_tokens=vit_num_tokens,
                num_queries=vit_num_tokens // 4,
                visual_hidden_size=visual_hidden_size,
                llm_hidden_size=self.llm.config.hidden_size)

            self.projector_num_queries = [vit_num_tokens // 4]

            print_log(f'set honeybee projector from {cfg_pth}', 'current')
            self.projector = HoneybeeProjectorModel(projector_config).to(self.visual_encoder.dtype)
            print_log(f'set projector config: {projector_config}', 'current') 
            print_log(f'projector arch: {self.projector}', 'current')
        # # <==== HonybeeProjectorModel end 

        # ====> DualPathProjectorModel
        if projector_type == 'dualpath':
            cfg_pth = '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/honeybee_configs/Honeybee-C-7B-M144-config.json'

            if self.visual_encoder_name == "RADIOModel":
                visual_hidden_size = self.visual_encoder.config.args['teachers'][-1]['input_size']

                if dynamic_image_size is not None:
                    vit_num_tokens = (dynamic_image_size // self.visual_encoder.patch_size) ** 2
                else:
                    vit_num_tokens = (self.visual_encoder.preferred_resolution[0] // self.visual_encoder.patch_size) ** 2
            else:
                visual_hidden_size = self.visual_encoder.config.hidden_size

                if dynamic_image_size is not None:
                    vit_num_tokens = (dynamic_image_size // self.visual_encoder.config.patch_size) ** 2
                else:
                    vit_num_tokens = (self.visual_encoder.config.image_size // self.visual_encoder.config.patch_size) ** 2

            if self.dino_visual_encoder:
                visual_hidden_size += self.dino_visual_encoder.config.hidden_size
            if self.convnext_visual_encoder:
                # visual_hidden_size += self.convnext_visual_encoder.feature_info[-1]['num_chs']
                visual_hidden_size += self.convnext_visual_encoder.config.hidden_sizes[-1]


            config_thumbnail = dict(
                config_path=cfg_pth,
                vit_num_tokens=vit_num_tokens,
                num_queries=vit_num_tokens // 4,
                visual_hidden_size=visual_hidden_size,
                llm_hidden_size=self.llm.config.hidden_size)

            config_tiles = dict(
                config_path=cfg_pth,
                vit_num_tokens=vit_num_tokens,
                num_queries=vit_num_tokens // 16,
                visual_hidden_size=visual_hidden_size,
                llm_hidden_size=self.llm.config.hidden_size)


            projector_config = DualPathProjectorConfig(
                config_thumbnail=config_thumbnail,
                config_tiles=config_tiles
            )

            self.projector_num_queries = [vit_num_tokens // 4, vit_num_tokens // 16]

            self.projector = DualPathProjectorModel(projector_config).to(self.visual_encoder.dtype)
            print_log(f'set projector config: {projector_config}', 'current') 
            print_log(f'projector arch: {self.projector}', 'current')
        # <==== DualPathProjectorModel end 


        # # # ====> CascadedHoneybeeProjectorModel
        # if projector_type == 'cascadedhoneybee':
        #     cfg_pth = '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/honeybee_configs/Honeybee-C-7B-M144-config.json'

        #     if self.visual_encoder_name == "RADIOModel":
        #         visual_hidden_size = self.visual_encoder.config.args['teachers'][-1]['input_size']

        #         if dynamic_image_size is not None:
        #             vit_num_tokens = (dynamic_image_size // self.visual_encoder.patch_size) ** 2
        #         else:
        #             vit_num_tokens = (self.visual_encoder.preferred_resolution[0] // self.visual_encoder.patch_size) ** 2
        #     else:
        #         visual_hidden_size = self.visual_encoder.config.hidden_size

        #         if dynamic_image_size is not None:
        #             vit_num_tokens = (dynamic_image_size // self.visua
        # l_encoder.config.patch_size) ** 2
        #         else:
        #             vit_num_tokens = (self.visual_encoder.config.image_size // self.visual_encoder.config.patch_size) ** 2

        #     if self.dino_visual_encoder:
        #         visual_hidden_size += self.dino_visual_encoder.config.hidden_size
        #     if self.convnext_visual_encoder:
        #         # visual_hidden_size += self.convnext_visual_encoder.feature_info[-1]['num_chs']
        #         visual_hidden_size += self.convnext_visual_encoder.config.hidden_sizes[-1]

        #     projector_config = HoneybeeProjectorConfig(
        #         config_path=cfg_pth,
        #         vit_num_tokens=vit_num_tokens,
        #         num_queries=vit_num_tokens // 4,
        #         visual_hidden_size=visual_hidden_size,
        #         llm_hidden_size=self.llm.config.hidden_size,
        #         cascaded=True,
        #         num_sub_images=num_sub_images)

        #     self.projector_num_queries = [vit_num_tokens // 4, vit_num_tokens // 16]

        #     print_log(f'set honeybee projector from {cfg_pth}', 'current')
        #     self.projector = HoneybeeProjectorModel(projector_config).to(self.visual_encoder.dtype)
        #     print_log(f'set projector config: {projector_config}', 'current') 
        #     print_log(f'projector arch: {self.projector}', 'current')
        # # # <==== DualHoneybeeProjectorModel end

        # # ====> TokenPackerProjectorModel
        if projector_type == 'tokenpacker':
            if dynamic_image_size is not None:
                raw_grid = dynamic_image_size // self.visual_encoder.config.patch_size
            else:
                raw_grid = self.visual_encoder.config.image_size // self.visual_encoder.config.patch_size
            
            visual_hidden_size = self.visual_encoder.config.hidden_size
            if self.dino_visual_encoder:
                visual_hidden_size += self.dino_visual_encoder.config.hidden_size

            projector_config = TokenPackerConfig(
                visual_hidden_size=visual_hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                raw_grid=raw_grid,
                scale_factor=4
            )

            self.projector = TokenPackerProjectorModel(projector_config).to(self.visual_encoder.dtype)
            print_log(f'set projector config: {projector_config}', 'current') 
            print_log(f'projector arch: {self.projector}', 'current')
        # # <==== TokenPackerProjectorModel end 

        # # ====> LDPProjectorModel
        if projector_type == 'ldp':
            projector_config = LDPProjectorConfig(
                visual_hidden_size=self.visual_encoder.config.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size, 
                # dw_size=(16, 16)
            )
            self.projector = LDPProjectorModel(projector_config).to(self.visual_encoder.dtype)
            print_log(f'set projector config: {projector_config}', 'current') 
            print_log(f'projector arch: {self.projector}', 'current')
        # # <==== LDPProjectorModel end 

        # # # ====> UHDProjectorModel
        # if projector_type == 'ca':
        #     if dynamic_image_size is not None:
        #         patches = dynamic_image_size // self.visual_encoder.config.patch_size
        #         tgt_size = (patches, patches)
        #     else:
        #         patches = self.visual_encoder.config.image_size // self.visual_encoder.config.patch_size
        #         tgt_size = (patches, patches)

        #     print_log(f'tgt_size {tgt_size} in UHDProjectorModel', 'current')

        #     projector_config = UHDProjectorConfig(
        #         kv_dim=self.visual_encoder.config.hidden_size,
        #         embed_dim=self.llm.config.hidden_size,
        #         tgt_size=tgt_size)
        #     self.projector = UHDProjectorModel(projector_config).to(self.visual_encoder.dtype)   
        #     print_log(f'set projector config: {projector_config}', 'current') 
        #     print_log(f'projector arch: {self.projector}', 'current')
        # # # <==== UHDProjectorModel end 

        self.vote_and_mix = False
        # # ====> FusionProjectorModel
        if projector_type == 'fusion':
            
            if self.visual_encoder_name == "RADIOModel":
                visual_hidden_size = self.visual_encoder.config.args['teachers'][-1]['input_size']

                if self.fusion_channel:
                    channel_fusion_hs1 = self.visual_encoder.config.args['teachers'][-1]['input_size']
                    visual_hidden_size += self.visual_encoder.config.args['teachers'][-1]['input_size']
                else:
                    channel_fusion_hs1 = None
                    channel_fusion_hs2 = None
                    
                if dynamic_image_size is not None:
                    vit_num_tokens = (dynamic_image_size // self.visual_encoder.patch_size) ** 2
                else:
                    vit_num_tokens = (self.visual_encoder.preferred_resolution[0] // self.visual_encoder.patch_size) ** 2
            else:
                visual_hidden_size = self.visual_encoder.config.hidden_size

                if self.fusion_channel:
                    channel_fusion_hs1 = self.visual_encoder.config.hidden_size
                    visual_hidden_size += self.visual_encoder.config.hidden_size
                else:
                    channel_fusion_hs1 = None
                    channel_fusion_hs2 = None

                if dynamic_image_size is not None:
                    vit_num_tokens = (dynamic_image_size // self.visual_encoder.config.patch_size) ** 2
                else:
                    vit_num_tokens = (self.visual_encoder.config.image_size // self.visual_encoder.config.patch_size) ** 2

            if self.dino_visual_encoder:
                visual_hidden_size += self.dino_visual_encoder.config.hidden_size

            if self.convnext_visual_encoder:
                visual_hidden_size += self.convnext_visual_encoder.feature_info[-1]['num_chs']
                # visual_hidden_size += self.convnext_visual_encoder.config.hidden_sizes[-1]

                if self.fusion_channel:
                    channel_fusion_hs2 = self.convnext_visual_encoder.feature_info[-1]['num_chs']
                    # channel_fusion_hs2 = self.convnext_visual_encoder.config.hidden_sizes[-1]
                    
            print_log(f'vit_num_tokens {vit_num_tokens} in FusionProjectorModel', 'current')
            
            scale_factor = [4, 16]
            hws = [vit_num_tokens // x for x in scale_factor]

            projector_config = FusionProjectorConfig(
                visual_hidden_size=visual_hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                # hidden_size=20480,
                vit_num_tokens=vit_num_tokens,
                hws=hws,
                projector_type='',
                channel_fusion=self.fusion_channel,
                channel_fusion_hs1=channel_fusion_hs1,
                channel_fusion_hs2=channel_fusion_hs2,
                # mlp_depth=4,
                # num_reg_queries=vit_num_tokens // 16,
                # num_ca_queries=128,
                # num_feats=vit_num_tokens // 4
            )

            # self.projector_num_queries = [vit_num_tokens // 4, 128]
            self.vote_and_mix = (projector_config.projector_type == 'vote_and_mix')
            self.projector_num_queries = hws

            print_log(f"projector_num_queries: {self.projector_num_queries}", 'current')

            self.projector = FusionProjectorModel(projector_config).to(self.visual_encoder.dtype)   
            print_log(f'set projector config: {projector_config}', 'current') 
            print_log(f'projector arch: {self.projector}', 'current')

            # print_log(f"Freeze part modules of FusionProjectorModel", "current")
            # for n, p in self.projector.named_parameters():
            #     if 'sub_net' in n: continue
            #     p.requires_grad = False

        # # <==== FusionProjectorModel end 

        # # ====> Idefics2ProjectorModel
        if projector_type == 'idefics2':
            projector_config = Idefics2ProjectorConfig(
                vision_config=self.visual_encoder.config,
                text_config=self.llm.config
            )
            self.projector = Idefics2ProjectorModel(projector_config).to(self.visual_encoder.dtype)
            print_log(f'set projector config: {projector_config}', 'current') 
            print_log(f'projector arch: {self.projector}', 'current')
        # # <==== Idefics2ProjectorModel end

        # ====> InternVL1.5ProjectorModel
        if projector_type == 'pixel_shuffle':

            if self.visual_encoder_name == "RADIOModel":
                visual_hidden_size = self.visual_encoder.config.args['teachers'][-1]['input_size']
            else:
                visual_hidden_size = self.visual_encoder.config.hidden_size

            projector_config = ProjectorConfig(
                visual_hidden_size=visual_hidden_size * 4,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=projector_depth,
                pixel_shuffle=True)
            self.projector = ProjectorModel(projector_config).to(self.visual_encoder.dtype)
            print_log(f'set projector config: {projector_config}', 'current') 
            print_log(f'projector arch: {self.projector}', 'current')
        # <==== InternVL1.5ProjectorModel end 

        # ====> ProjectorModel
        if projector_type == 'mlp':

            if self.visual_encoder_name == "RADIOModel":
                visual_hidden_size = self.visual_encoder.config.args['teachers'][-1]['input_size']
            else:
                visual_hidden_size = self.visual_encoder.config.hidden_size
            
            if self.dino_visual_encoder:
                visual_hidden_size = self.visual_encoder.config.hidden_size+self.dino_visual_encoder.config.hidden_size
            else:
                visual_hidden_size = self.visual_encoder.config.hidden_size

            projector_config = ProjectorConfig(
                visual_hidden_size=visual_hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=projector_depth)
            self.projector = ProjectorModel(projector_config).to(self.visual_encoder.dtype)
            print_log(f'set projector config: {projector_config}', 'current') 
            print_log(f'projector arch: {self.projector}', 'current')
        # <==== ProjectorModel end 

        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)
            if dino_visual_encoder is not None:
                self.dino_visual_encoder.requires_grad_(False)
            if convnext_visual_encoder is not None:
                self.convnext_visual_encoder.requires_grad_(False)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
            if hasattr(self.visual_encoder, 'enable_input_require_grads'):
                try:
                    self.visual_encoder.enable_input_require_grads()
                except:
                    print_log(f"{self.visual_encoder.__class__.__name__} does not implement enable_input_require_grads.", "current")
            else:
                self.visual_encoder.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)
            self.projector.enable_input_require_grads()

            if dino_visual_encoder is not None:
                if hasattr(self.dino_visual_encoder, 'enable_input_require_grads'):
                    self.dino_visual_encoder.enable_input_require_grads()
                else:
                    self.dino_visual_encoder.get_input_embeddings(
                    ).register_forward_hook(make_inputs_require_grad)

            if convnext_visual_encoder is not None:
                if hasattr(self.convnext_visual_encoder, 'enable_input_require_grads'):
                    try:
                        self.convnext_visual_encoder.enable_input_require_grads()
                    except:
                        print_log(f"{self.convnext_visual_encoder} does not implement enable_input_require_grads.", "current")
                else:
                    self.convnext_visual_encoder.register_forward_hook(make_inputs_require_grad)

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None

        if self.use_llm_lora:
            print_log(f'use llm lora', 'current')
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing, llm_lora_adapt_mode)
        if self.use_visual_encoder_lora:
            print_log(f'use visual encoder lora for {self.visual_encoder.__class__.__name__}', 'current')
            self._prepare_visual_encoder_for_lora(
                visual_encoder_lora, use_activation_checkpointing)
            
            if self.dino_visual_encoder is not None:
                print_log(f'use visual encoder lora for {self.dino_visual_encoder.__class__.__name__}', 'current')
                self._prepare_dino_visual_encoder_for_lora(
                    visual_encoder_lora, use_activation_checkpointing)
            
            if self.convnext_visual_encoder is not None and self.activate_convnext_lora:
                print_log(f'use visual encoder lora for {self.convnext_visual_encoder.__class__.__name__}', 'current')
                self._prepare_convnext_visual_encoder_for_lora(
                        visual_encoder_lora, use_activation_checkpointing)

        if self.freeze_visualenc_lora:
            self.visual_encoder.requires_grad_(False)
            if dino_visual_encoder is not None:
                self.dino_visual_encoder.requires_grad_(False)
            if convnext_visual_encoder is not None:
                self.convnext_visual_encoder.requires_grad_(False)
                
        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            new_pretrained_state_dict = {}

            for k, v in pretrained_state_dict.items():

                if 'projector.abstractor.net.' in k:
                    new_key = k.replace('abstractor', 'hb1')
                    new_pretrained_state_dict[new_key] = v

                elif 'projector.abstractor.pos_emb' in k or 'projector.abstractor.readout.' in k:
                    new_key = k.replace('abstractor.', '')
                    new_pretrained_state_dict[new_key] = v
                
                else:
                    new_pretrained_state_dict[k] = v

            # new_pretrained_state_dict.update(pretrained_state_dict)

            self.load_state_dict(new_pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}', 'current')

        print_log(f'Projector total parameters: {sum(p.numel() for p in self.projector.parameters())}, trainable parameters: {sum(p.numel() for p in self.projector.parameters() if p.requires_grad)}', 'current') 
        print_log(f'visual_encoder total parameters: {sum(p.numel() for p in self.visual_encoder.parameters())}, trainable parameters: {sum(p.numel() for p in self.visual_encoder.parameters() if p.requires_grad)}', 'current') 
        if dino_visual_encoder is not None:
            print_log(f'dino_visual_encoder total parameters: {sum(p.numel() for p in self.dino_visual_encoder.parameters())}, trainable parameters: {sum(p.numel() for p in self.dino_visual_encoder.parameters() if p.requires_grad)}', 'current') 
        if convnext_visual_encoder is not None:
            print_log(f'convnext_visual_encoder total parameters: {sum(p.numel() for p in self.convnext_visual_encoder.parameters())}, trainable parameters: {sum(p.numel() for p in self.convnext_visual_encoder.parameters() if p.requires_grad)}', 'current') 

        self._is_init = True

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True,
                              adapt_mode=None):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            if adapt_mode == 'lm_head':
                print_log('only adapt lora for lm head', 'current')
                modules = find_lm_head_names(self.llm)
            elif adapt_mode == "llm_with_lm_head":
                print_log('adapt lora for llm and lm_head', 'current')
                modules = find_all_linear_names_with_lm_head(self.llm)
            else:
                print_log('adapt lora for llm, except lm_head', 'current')
                modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_visual_encoder_for_lora(self,
                                         lora_config,
                                         use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.visual_encoder)
            # modules = find_all_linear_conv_names(self.visual_encoder)
            lora_config.target_modules = modules
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

    def _prepare_dino_visual_encoder_for_lora(self,
                                              lora_config,
                                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.dino_visual_encoder)
            # modules = find_all_linear_conv_names(self.dino_visual_encoder)
            lora_config.target_modules = modules
        self.dino_visual_encoder = get_peft_model(self.dino_visual_encoder, lora_config)

    def _prepare_convnext_visual_encoder_for_lora(self,
                                              lora_config,
                                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_conv_names(self.convnext_visual_encoder)
            print_log(f"convnext lora modules: {modules}", "current")
            lora_config.target_modules = modules
        self.convnext_visual_encoder = get_peft_model(self.convnext_visual_encoder, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        try:
            self.visual_encoder.gradient_checkpointing_enable()
        except:
            print_log(f"{self.visual_encoder.__class__.__name__} does not support gradient checkpointing.", "current")
        self.projector.gradient_checkpointing_enable()
        if self.dino_visual_encoder is not None:
            self.dino_visual_encoder.gradient_checkpointing_enable()
        if self.convnext_visual_encoder is not None:
            try:
                self.convnext_visual_encoder.set_grad_checkpointing(enable=True)
            except:
                print_log(f'{self.convnext_visual_encoder} does not support gradient checkpointing.', 'current')

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.visual_encoder.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()
        if self.dino_visual_encoder is not None:
            self.dino_visual_encoder.gradient_checkpointing_disable()
        if self.convnext_visual_encoder is not None:
            try:
                self.convnext_visual_encoder.set_grad_checkpointing(enable=False)
            except:
                print_log(f'{self.convnext_visual_encoder} does not support gradient checkpointing.', 'current')

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.visual_encoder, state_dict=state_dict))
            if self.dino_visual_encoder:
                to_return.update(
                    get_peft_model_state_dict(
                        self.dino_visual_encoder, state_dict=state_dict))
            if self.convnext_visual_encoder and self.activate_convnext_lora:
                to_return.update(
                    get_peft_model_state_dict(
                        self.convnext_visual_encoder, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
            
            if self.dino_visual_encoder:
                to_return.update({
                    k: v
                    for k, v in state_dict.items() if 'dino_visual_encoder.' in k
                })

            if self.convnext_visual_encoder:
                to_return.update({
                    k: v
                    for k, v in state_dict.items() if 'convnext_visual_encoder.' in k
                })

        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})
        return to_return

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            print(f"cfg_or_mod is nn.Module")
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            print(f"cfg_or_mod is dict")
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError
        
    def visual_feature_select(self, image_forward_outs, layers=None, with_cls=True):
        if layers is not None:
            image_feature_list = []
            for l in layers:
                image_feature_list.append(image_forward_outs.hidden_states[l][:, 1:] if with_cls else image_forward_outs.hidden_states[l])
            image_features_multi = torch.cat(image_feature_list, dim=-1)
        else:
            image_features_multi = image_forward_outs.hidden_states[self.visual_select_layer][:, 1:] if with_cls else image_forward_outs.hidden_states[self.visual_select_layer]

        image_features = image_forward_outs.hidden_states[self.visual_select_layer][:, 1:] if with_cls else image_forward_outs.hidden_states[self.visual_select_layer]

        return (image_features, image_features_multi)

    def _forward_visual_encoders(self, input_image, convnext_input_image=None, fusion_channel=False):
        if input_image.dtype != self.visual_encoder.dtype:
            input_image = input_image.to(dtype=self.visual_encoder.dtype)

        if self.visual_encoder_name == "RADIOModel":
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                visual_outputs = self.visual_encoder(input_image)
        else:
            visual_outputs = self.visual_encoder(
                input_image, output_hidden_states=True)

        if self.dino_visual_encoder:
            # if data.get('dino_pixel_values', None) is not None:
            #     if data['dino_pixel_values'].dtype != self.dino_visual_encoder.dtype:
            #         data['dino_pixel_values'] = data['dino_pixel_values'].to(dtype=self.dino_visual_encoder.dtype)

            #     dino_outputs = self.dino_visual_encoder(
            #             data['dino_pixel_values'], output_hidden_states=True)
            # else:
            dino_outputs = self.dino_visual_encoder(
                    input_image, output_hidden_states=True)
                
        if self.convnext_visual_encoder and convnext_input_image is not None:
            if convnext_input_image.dtype != self.convnext_visual_encoder.stem[0].weight.dtype:
                convnext_input_image = convnext_input_image.to(dtype=self.convnext_visual_encoder.stem[0].weight.dtype)   

            convnext_outputs = self.convnext_visual_encoder.stem(convnext_input_image)
            for stage in self.convnext_visual_encoder.stages:
                convnext_outputs = stage(convnext_outputs)

        visual_feat_list = []

        if self.visual_encoder_name == "RADIOModel":
            vit_pixel_values = visual_outputs[-1].to(self.projector.dtype)
            # print(f"vit_pixel_values: {vit_pixel_values.shape}")
            visual_feat_list.append(vit_pixel_values)
        else:
            vit_pixel_values = visual_outputs.hidden_states[self.visual_select_layer]
            # print(f"vit_pixel_values: {vit_pixel_values.shape}")
            visual_feat_list.append(vit_pixel_values)

        if self.dino_visual_encoder:
            dino_pixel_values = dino_outputs.hidden_states[self.visual_select_layer][:, 1:]
            # print(f"dino_pixel_values: {dino_pixel_values.shape}")
            visual_feat_list.append(dino_pixel_values)

        if self.convnext_visual_encoder:
            # bs = vit_pixel_values.shape[0]
            # convnext_pixel_values = convnext_outputs.view(bs, convnext_outputs.shape[1], -1).permute(0, 2, 1)
            convnext_pixel_values = rearrange(convnext_outputs, "b d h w -> b (h w) d")
            # print(f"convnext_pixel_values: {convnext_pixel_values.shape}")
            visual_feat_list.append(convnext_pixel_values)
        
        if fusion_channel:
            visual_feats = self.projector.vit_channel_fusion(visual_feat_list)
        else:
            visual_feats = torch.cat(visual_feat_list, dim=-1)

        return visual_feats


    def forward(self, data, data_samples=None, mode='loss'):

        if 'pixel_values' in data:
    
            # print(f"data['pixel_values']: {data['pixel_values'].shape}")
            # print(f"data['pixel_values_tiles']: {data['pixel_values_tiles'].shape}")
            
            if 'pixel_values_tiles' in data:

                bs = data['pixel_values'].shape[0]

                # print(f"bs: {bs}")

                if data.get('convnext_pixel_values', None) is not None and data.get('convnext_pixel_values_tiles', None) is not None:
                    pixel_values = self._forward_visual_encoders(
                        torch.cat([data['pixel_values'], data['pixel_values_tiles']], dim=0), 
                        torch.cat([data['convnext_pixel_values'], data['convnext_pixel_values_tiles']], dim=0), 
                        self.fusion_channel)
                else:
                    pixel_values = self._forward_visual_encoders(torch.cat([data['pixel_values'], data['pixel_values_tiles']], dim=0), self.fusion_channel)
                
                if self.projector_type == 'dualpath' or self.projector_type == 'fusion':
                    thumbnail_pixel_values = pixel_values[:bs, ...]
                    tiles_pixel_values = pixel_values[bs:, ...]
                    # print(f"thumbnail_pixel_values: {thumbnail_pixel_values.shape}")
                    # print(f"tiles_pixel_values: {tiles_pixel_values.shape}")
                    visual_feat_list = [thumbnail_pixel_values, tiles_pixel_values]
                else:
                    visual_feat_list = [pixel_values]

            else:
                bs = data['pixel_values'].shape[0] // data['max_tiles']

                # print(f"bs: {bs}")

                pixel_values = self._forward_visual_encoders(data['pixel_values'])

                visual_feat_list = [pixel_values]

            if len(visual_feat_list) > 1:
                pixel_values = self.projector(visual_feat_list)
            else:
                # pixel_values = self.projector((visual_feat_list[0], (x // self.visual_encoder.patch_size for x in data['image_size'])))
                pixel_values = self.projector(visual_feat_list[0])

            # print(f"pixel_values: {pixel_values.shape}")

            # print(f"data['num_pixel_values']: {data['num_pixel_values']}")
            # print(f"data['max_tiles']: {data['max_tiles']}")

            if self.num_sub_images > 1 and (not self.vote_and_mix):

                # if self.projector_type != "dualpath":
                #     thumbnail_pixel_values = pixel_values[:bs, ...]
                #     tiles_pixel_values = pixel_values[bs:, ...].view(bs, -1, pixel_values.shape[-1])
                #     pixel_values = torch.cat([thumbnail_pixel_values, tiles_pixel_values], dim=1)
                
                if 'pixel_values_tiles' not in data:  
                    pixel_values = pixel_values.view(bs, -1, pixel_values.shape[2])

                data['pixel_values'] = pixel_values
                data['projector_num_queries'] = self.projector_num_queries

                # print(f"{'=' * 50} rank {dist.get_rank()} LLaVAModel forward {'=' * 50}")
                # print(f"data['data_ids']: {data['data_ids']}")
                # print(f"data['pixel_values']: {data['pixel_values'].shape}")
                # print(f"data['projector_num_queries']: {data['projector_num_queries']}")
                # print(f"data['num_pixel_values']: {data['num_pixel_values']}")

                data = prepare_multi_subimages_inputs_labels_for_multimodal(llm=self.llm, llm_name=self.llm_name, 
                                                                            add_img_slice_special_tokens=self.add_img_slice_special_tokens, mark_slice_indices=self.mark_slice_indices, 
                                                                            **data)
            else:
                data['pixel_values'] = pixel_values
                data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)

        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        # print(f"outputs: {outputs}")
        # quit()
        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
