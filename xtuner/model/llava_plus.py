# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
import torch.nn as nn
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
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, prepare_multi_subimages_inputs_labels_for_multimodal,
                    traverse_dict)

from transformers import AutoModel, AutoConfig, SiglipVisionModel, ConvNextConfig, ConvNextModel
import open_clip
from safetensors import safe_open

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


class LLaVAPlusModel(BaseModel):

    def __init__(self,
                 llm,
                 visual_encoder,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_type='mlp',
                 projector_depth=2,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 siglip_visual_encoder=None,
                 dino_visual_encoder=None,
                 convnext_visual_encoder=None,
                 use_activation_checkpointing=True,
                 dynamic_image_size=None,
                 num_sub_images=0,
                 ):
        super().__init__()

        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.visual_select_layer = visual_select_layer
        self.num_sub_images = num_sub_images

        with LoadWoInit():
            self.llm = self._build_from_cfg_or_module(llm)

            self.visual_encoder = self._build_from_cfg_or_module(visual_encoder)

            if siglip_visual_encoder is not None:
                self.siglip_visual_encoder = self._build_siglip_visual_encoder(siglip_visual_encoder, dynamic_image_size)
            else:
                self.siglip_visual_encoder = None

            if dino_visual_encoder is not None:
                self.dino_visual_encoder = self._build_from_cfg_or_module(dino_visual_encoder)
            else:
                 self.dino_visual_encoder = None

            if convnext_visual_encoder is not None:
                # self.convnext_visual_encoder = open_clip.create_model(convnext_visual_encoder[0], pretrained=convnext_visual_encoder[1]).visual.trunk
                self.convnext_visual_encoder = self._build_from_cfg_or_module(convnext_visual_encoder)
            else:
                 self.convnext_visual_encoder = None

            self.visual_encoder_name = self.visual_encoder.__class__.__name__
            self.siglip_visual_encoder_name = self.siglip_visual_encoder.__class__.__name__ if self.siglip_visual_encoder else None
            self.dino_visual_encoder_name = self.dino_visual_encoder.__class__.__name__ if self.dino_visual_encoder else None
            self.convnext_visual_encoder_name = self.convnext_visual_encoder.__class__.__name__ if self.convnext_visual_encoder else None
            
            print_log(f"build visual_encoder {self.visual_encoder_name} ", 'current')
            print_log(f"build dino_visual_encoder: {self.dino_visual_encoder_name} ", 'current')
            print_log(f"build convnext_visual_encoder: {self.convnext_visual_encoder_name} ", 'current')

        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        self.projector_type = projector_type

        # # ====> CascadedHoneybeeProjectorModel
        if projector_type == 'cascadedhoneybee':
            cfg_pth = '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/honeybee_configs/Honeybee-C-7B-M144-config.json'

            visual_hidden_size = self.visual_encoder.config.args['teachers'][-1]['input_size']
            vit_num_tokens = (self.visual_encoder.preferred_resolution[0] // self.visual_encoder.patch_size) ** 2


            if self.siglip_visual_encoder:
                sub_visual_hidden_size = self.siglip_visual_encoder.config.hidden_size

                if dynamic_image_size is not None:
                    sub_vit_num_tokens = (dynamic_image_size // self.siglip_visual_encoder.config.patch_size) ** 2
                else:
                    sub_vit_num_tokens = (self.siglip_visual_encoder.config.image_size // self.siglip_visual_encoder.config.patch_size) ** 2

            # if self.dino_visual_encoder:
            #     visual_hidden_size += self.dino_visual_encoder.config.hidden_size
            # if self.convnext_visual_encoder:
            #     # visual_hidden_size += self.convnext_visual_encoder.feature_info[-1]['num_chs']
            #     visual_hidden_size += self.convnext_visual_encoder.config.hidden_sizes[-1]

            projector_config = HoneybeeProjectorConfig(
                config_path=cfg_pth,
                vit_num_tokens=vit_num_tokens,
                num_queries=vit_num_tokens // 4,
                visual_hidden_size=visual_hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,)

            sub_projector_config = HoneybeeProjectorConfig(
                config_path=cfg_pth,
                vit_num_tokens=sub_vit_num_tokens,
                num_queries=sub_vit_num_tokens // 4,
                visual_hidden_size=sub_visual_hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,)

            self.projector_num_queries = [vit_num_tokens // 4, sub_vit_num_tokens // 4]

            print_log(f'set honeybee projector from {cfg_pth}', 'current')

            self.projector = HoneybeeProjectorModel(projector_config).to(self.visual_encoder.dtype)
            self.sub_projector = HoneybeeProjectorModel(sub_projector_config).to(self.visual_encoder.dtype)

            print_log(f'set projector config: {projector_config}', 'current') 
            print_log(f'projector arch: {self.projector}', 'current')

            print_log(f'set sub_projector config: {sub_projector_config}', 'current') 
            print_log(f'sub projector arch: {self.sub_projector}', 'current')
        # # <==== DualHoneybeeProjectorModel end

        # ====> InternVL1.5DualPathProjectorModel
        if projector_type == 'dualpath':
            cfg_pth = '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/honeybee_configs/Honeybee-C-7B-M144-config.json'

            if self.visual_encoder_name == "RADIOModel":
                visual_hidden_size1 = self.visual_encoder.config.args['teachers'][-1]['input_size']
            else:
                visual_hidden_size1 = self.visual_encoder.config.hidden_size

            if self.siglip_visual_encoder:

                visual_hidden_size2 = self.siglip_visual_encoder.config.hidden_size

                if dynamic_image_size is not None:
                    vit_num_tokens = (dynamic_image_size // self.siglip_visual_encoder.config.patch_size) ** 2
                else:
                    vit_num_tokens = (self.siglip_visual_encoder.config.image_size // self.siglip_visual_encoder.config.patch_size) ** 2


            config_thumbnail = ProjectorConfig(
                visual_hidden_size=visual_hidden_size1 * 4,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=projector_depth,
                pixel_shuffle=True)
            config_tiles = HoneybeeProjectorConfig(
                config_path=cfg_pth,
                vit_num_tokens=vit_num_tokens,
                num_queries=vit_num_tokens // 4,
                visual_hidden_size=visual_hidden_size2,
                llm_hidden_size=self.llm.config.hidden_size)


            projector_config = DualPathProjectorConfig(
                config_thumbnail=config_thumbnail,
                config_tiles=config_tiles
            )

            self.projector = DualPathProjectorModel(projector_config).to(self.visual_encoder.dtype)
            print_log(f'set projector config: {projector_config}', 'current') 
            print_log(f'projector arch: {self.projector}', 'current')
        # <==== InternVL1.5DualPathProjectorModel end 


        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)
            if siglip_visual_encoder is not None:
                self.siglip_visual_encoder.requires_grad_(False)
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

            if siglip_visual_encoder is not None:
                if hasattr(self.siglip_visual_encoder, 'enable_input_require_grads'):
                    self.siglip_visual_encoder.enable_input_require_grads()
                else:
                    self.siglip_visual_encoder.get_input_embeddings(
                    ).register_forward_hook(make_inputs_require_grad)

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
                        print_log("ConvNextModel does not implement enable_input_require_grads.", "current")
                else:
                    self.convnext_visual_encoder.get_input_embeddings(
                    ).register_forward_hook(make_inputs_require_grad)

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None

        if self.use_llm_lora:
            print_log(f'use llm lora', 'current')
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
        if self.use_visual_encoder_lora:
            print_log(f'use visual encoder lora for {self.visual_encoder.__class__.__name__}', 'current')
            self.visual_encoder = self._prepare_visual_encoder_for_lora(
                visual_encoder_lora, self.visual_encoder, use_activation_checkpointing)

            if self.siglip_visual_encoder is not None:
                print_log(f'use visual encoder lora for {self.siglip_visual_encoder.__class__.__name__}', 'current')
                self.siglip_visual_encoder = self._prepare_visual_encoder_for_lora(
                    visual_encoder_lora, self.siglip_visual_encoder, use_activation_checkpointing)

            if self.dino_visual_encoder is not None:
                print_log(f'use visual encoder lora for {self.dino_visual_encoder.__class__.__name__}', 'current')
                self.dino_visual_encoder = self._prepare_visual_encoder_for_lora(
                    visual_encoder_lora, self.dino_visual_encoder, use_activation_checkpointing)

            # if self.convnext_visual_encoder is not None:
            #     print_log(f'use visual encoder lora for {self.convnext_visual_encoder.__class__.__name__}', 'current')
            #     self._prepare_convnext_visual_encoder_for_lora(
            #         visual_encoder_lora, use_activation_checkpointing)
                
        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}', 'current')

        print_log(f'Projector total parameters: {sum(p.numel() for p in self.projector.parameters())}, trainable parameters: {sum(p.numel() for p in self.projector.parameters() if p.requires_grad)}', 'current') 
        print_log(f'visual_encoder total parameters: {sum(p.numel() for p in self.visual_encoder.parameters())}, trainable parameters: {sum(p.numel() for p in self.visual_encoder.parameters() if p.requires_grad)}', 'current') 
        if siglip_visual_encoder is not None:
            print_log(f'siglip_visual_encoder total parameters: {sum(p.numel() for p in self.siglip_visual_encoder.parameters())}, trainable parameters: {sum(p.numel() for p in self.siglip_visual_encoder.parameters() if p.requires_grad)}', 'current')         
        if dino_visual_encoder is not None:
            print_log(f'dino_visual_encoder total parameters: {sum(p.numel() for p in self.dino_visual_encoder.parameters())}, trainable parameters: {sum(p.numel() for p in self.dino_visual_encoder.parameters() if p.requires_grad)}', 'current') 
        if convnext_visual_encoder is not None:
            print_log(f'convnext_visual_encoder total parameters: {sum(p.numel() for p in self.convnext_visual_encoder.parameters())}, trainable parameters: {sum(p.numel() for p in self.convnext_visual_encoder.parameters() if p.requires_grad)}', 'current') 


        self._is_init = True

    def _build_siglip_visual_encoder(self, siglip_visual_encoder, dynamic_image_size):
        if dynamic_image_size is None:
            visual_encoder = self._build_from_cfg_or_module(siglip_visual_encoder)
        else:
            print_log(f'build siglip vit model', 'current')
            visual_encoder_configuration = AutoConfig.from_pretrained(siglip_visual_encoder['pretrained_model_name_or_path']).vision_config
            visual_encoder = SiglipVisionModel(config=visual_encoder_configuration).to(dtype=siglip_visual_encoder['torch_dtype']) 

            vit_model_state_dict = {}
            with safe_open(f"{siglip_visual_encoder['pretrained_model_name_or_path']}/model.safetensors", framework="pt", device="cpu") as f:
                for k in f.keys():
                    if 'vision_model' in k: 
                        vit_model_state_dict[k] = f.get_tensor(k)

            visual_encoder.load_state_dict(vit_model_state_dict)

            print_log(f'interpolate pos embed for siglip vit model to {dynamic_image_size}', 'current')
            pos_embed = visual_encoder.vision_model.embeddings.position_embedding.weight
            pos_embed = pos_embed.float()
            embedding_size = pos_embed.shape[-1]
            orig_size = int(visual_encoder.vision_model.embeddings.image_size / visual_encoder.vision_model.embeddings.patch_size)
            new_size = int(dynamic_image_size / visual_encoder.vision_model.embeddings.patch_size)
            
            print_log("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size), 'current')

            pos_embed = pos_embed.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(
                pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
            position_embedding_weight = pos_embed.permute(0, 2, 3, 1).flatten(1, 2).squeeze()
            position_embedding_weight = position_embedding_weight.to(visual_encoder.dtype)

            num_patches = (dynamic_image_size // visual_encoder.vision_model.embeddings.patch_size) ** 2
            print_log(f'replace pos embed for siglip vit', 'current')
            num_positions = num_patches
            position_embedding = torch.nn.Embedding(num_positions, visual_encoder.vision_model.embeddings.embed_dim)
            visual_encoder.vision_model.embeddings.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent = False)
            position_embedding.weight = torch.nn.Parameter(position_embedding_weight)
            visual_encoder.vision_model.embeddings.position_embedding = position_embedding
        return visual_encoder


    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_visual_encoder_for_lora(self,
                                         lora_config,
                                         visual_encoder,
                                         use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(visual_encoder)
            lora_config.target_modules = modules
        visual_encoder = get_peft_model(visual_encoder, lora_config)
        return visual_encoder

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
                self.convnext_visual_encoder.gradient_checkpointing_enable()
            except:
                print_log('ConvNextModel does not support gradient checkpointing.', 'current')

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.visual_encoder.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()
        if self.siglip_visual_encoder is not None:
            self.siglip_visual_encoder.gradient_checkpointing_disable()
        if self.dino_visual_encoder is not None:
            self.dino_visual_encoder.gradient_checkpointing_disable()
        if self.convnext_visual_encoder is not None:
            self.convnext_visual_encoder.gradient_checkpointing_disable()

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
            if self.siglip_visual_encoder:
                to_return.update(
                    get_peft_model_state_dict(
                        self.siglip_visual_encoder, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })

            if self.siglip_visual_encoder:
                to_return.update({
                    k: v
                    for k, v in state_dict.items() if 'siglip_visual_encoder.' in k
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
        
    # def visual_feature_select(self, image_forward_outs, layers=None, with_cls=True):
    #     if layers is not None:
    #         image_feature_list = []
    #         for l in layers:
    #             image_feature_list.append(image_forward_outs.hidden_states[l][:, 1:] if with_cls else image_forward_outs.hidden_states[l])
    #         image_features_multi = torch.cat(image_feature_list, dim=-1)
    #     else:
    #         image_features_multi = image_forward_outs.hidden_states[self.visual_select_layer][:, 1:] if with_cls else image_forward_outs.hidden_states[self.visual_select_layer]

    #     image_features = image_forward_outs.hidden_states[self.visual_select_layer][:, 1:] if with_cls else image_forward_outs.hidden_states[self.visual_select_layer]

    #     return (image_features, image_features_multi)

    def forward(self, data, data_samples=None, mode='loss'):

        # for k, v in data.items():
        #     if v is not None:
        #         print(k, v.shape)
        #     else:
        #         print(k, "None")

        if 'pixel_values' in data:
            
            # print(f"input_ids: {data['input_ids']}")

            print(f"data['pixel_values']: {data['pixel_values'].shape}")
            print(f"data['pixel_values_tiles']: {data['pixel_values_tiles'].shape}")
        
            data['pixel_values'] = data['pixel_values'].to(dtype=self.visual_encoder.dtype)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                visual_outputs = self.visual_encoder(data['pixel_values'])

            if self.siglip_visual_encoder:
                data['pixel_values_tiles'] = data['pixel_values_tiles'].to(dtype=self.siglip_visual_encoder.dtype)
                siglip_visual_outputs = self.siglip_visual_encoder(
                    data['pixel_values_tiles'], output_hidden_states=True)

            visual_feat_list = []

            vit_pixel_values = visual_outputs[-1].to(self.projector.dtype)
            print(f"vit_pixel_values: {vit_pixel_values.shape}")
            if 'image_size' in data:
                visual_feat_list.append((vit_pixel_values, [wh // self.visual_encoder.patch_size for wh in data['image_size']]))
            else:
                visual_feat_list.append(vit_pixel_values)

            print(f"data['image_size']: {data['image_size']}, {visual_feat_list[0][1]}")
            
            if self.siglip_visual_encoder:
                siglip_pixel_values = siglip_visual_outputs.hidden_states[self.visual_select_layer]
                print(f"siglip_pixel_values: {siglip_pixel_values.shape}")
                visual_feat_list.append(siglip_pixel_values)


            if len(visual_feat_list) > 1:
                # pixel_values_tumbnail = self.projector(visual_feat_list[0])
                # pixel_values_tiles = self.sub_projector(visual_feat_list[1])
                # pixel_values_tiles = pixel_values_tiles.view(-1, self.num_sub_images * pixel_values_tiles.shape[1], pixel_values_tiles.shape[2])

                # print(f"pixel_values_tumbnail: {pixel_values_tumbnail.shape}")
                # print(f"pixel_values_tiles: {pixel_values_tiles.shape}")

                # pixel_values = torch.cat([pixel_values_tumbnail, pixel_values_tiles], dim=1)

                pixel_values = self.projector(visual_feat_list)
            else:
                pixel_values = self.projector(visual_feat_list[0])

            # print(f"pixel_values: {pixel_values.shape}")

            # feature_dict = {
            #     'input_pixel_values': data['pixel_values'],
            #     'visual_outputs': visual_outputs,
            #     'dino_outputs': dino_outputs,
            #     'vit_pixel_values': vit_pixel_values,
            #     'dino_pixel_values': dino_pixel_values,
            #     'after_proj_pixel_values': pixel_values
            # }

            # torch.save(feature_dict, '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/output_tensors/feature_dict.bin')
            # torch.save(self.visual_encoder.state_dict(), '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/output_tensors/visual_encoder.bin')
            # torch.save(self.dino_visual_encoder.state_dict(), '/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/output_tensors/dino_visual_encoder.bin')
            # quit()

            # if self.dino_visual_encoder and self.convnext_visual_encoder is None:
            #     if self.projector.__class__.__name__ == 'TokenPackerProjectorModel':
                    
            #         # vit_pixel_values = self.visual_feature_select(visual_outputs, layers=[12,16,22,23], with_cls=False)
            #         # dino_pixel_values = self.visual_feature_select(dino_outputs, layers=[12,16,22,23], with_cls=True)

            #         vit_pixel_values = self.visual_feature_select(visual_outputs, with_cls=False)
            #         dino_pixel_values = self.visual_feature_select(dino_outputs, with_cls=True)

            #         pixel_values = (
            #             torch.cat([vit_pixel_values[0], dino_pixel_values[0]], dim=-1),
            #             torch.cat([vit_pixel_values[1], dino_pixel_values[1]], dim=-1)
            #         )

            #         pixel_values = self.projector(pixel_values)
            #     else:
            #         vit_pixel_values = visual_outputs.hidden_states[self.visual_select_layer]
            #         dino_pixel_values = dino_outputs.hidden_states[self.visual_select_layer][:, 1:]
            #         pixel_values = self.projector(torch.cat([vit_pixel_values, dino_pixel_values], dim=-1))


            #     # if self.projector.__class__.__name__ == 'FusionProjectorModel':
            #     #     pixel_values = self.projector((torch.cat([vit_pixel_values, dino_pixel_values], dim=-1), data['num_pixel_values']))
            #     # else:
            #     #     pixel_values = self.projector(torch.cat([vit_pixel_values, dino_pixel_values], dim=-1))

            # elif self.dino_visual_encoder and self.convnext_visual_encoder:

            #     vit_pixel_values = visual_outputs.hidden_states[self.visual_select_layer]
            #     dino_pixel_values = dino_outputs.hidden_states[self.visual_select_layer][:, 1:]

            #     bs = vit_pixel_values.shape[0]
            #     convnext_pixel_values = convnext_outputs.view(bs, convnext_outputs.shape[1], -1).permute(0, 2, 1)

            #     if self.projector.__class__.__name__ == 'FusionProjectorModel':
            #         pixel_values = self.projector((torch.cat([vit_pixel_values, dino_pixel_values, convnext_pixel_values], dim=-1), data['num_pixel_values']))
            #     else:
            #         pixel_values = self.projector(torch.cat([vit_pixel_values, dino_pixel_values, convnext_pixel_values], dim=-1))

            # else:
            #     pixel_values = self.projector(
            #         visual_outputs.hidden_states[self.visual_select_layer][:, 1:]) 

            if self.num_sub_images > 1:
                # if self.projector_type != "cascadedhoneybee":
                #     pixel_values = pixel_values.view(-1, (self.num_sub_images + 1) * pixel_values.shape[1], pixel_values.shape[2])
                data['pixel_values'] = pixel_values
                print(f"data['pixel_values']: {data['pixel_values'].shape}")
                # quit()
                # data['projector_num_queries'] = self.projector_num_queries
                data['projector_num_queries'] = [visual_feat_list[0][0].shape[1] // 4, visual_feat_list[1].shape[1] // 4]
                print(f"data['projector_num_queries']: {data['projector_num_queries']}")
                print(f"data['num_pixel_values']: {data['num_pixel_values']}")
                data = prepare_multi_subimages_inputs_labels_for_multimodal(llm=self.llm, **data)
            else:
                data['pixel_values'] = pixel_values
                data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)

        # print(f"====> data: {data.keys()}, pixel_values in data: {'pixel_values' in data}")
        # for k, v in data.items():
        #     if v is not None:
        #         print(k, v.shape)
        #     else:
        #         print(k, "None")
        # print(f"attention_mask: {data['attention_mask'][0]}, labels: {data['labels'][0]}")        
        # print(f"====> data_samples: {data_samples}")

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
