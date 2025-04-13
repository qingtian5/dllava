# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from timm.models.regnet import RegStage
from timm.layers import LayerNorm2d
from timm.layers.conv_bn_act import ConvNormAct
from einops import rearrange
from functools import partial
import torch.nn.functional as F

from .configuration_tokenpacker import TokenPackerConfig
from .configuration_fusion_projector import FusionProjectorConfig
from .modeling_tokenpacer_projector import TokenPacker
# from .honeybee_projectors import build_mlp

from mmengine import print_log

def build_mlp(depth, hidden_size, output_hidden_size, act_func):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        # layers.append(nn.SiLU())
        layers.append(ACT2FN[act_func])
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)

def build_hb_pos_embeds(
    num_input_tokens: int, vision_hidden_size: int
):
    pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
    nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)

    return pos_emb

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, num_queries: int=None, num_feats: int=None, use_pos_embed=True):
        super().__init__()
        self.n_head = n_head
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # self.query_norm = nn.LayerNorm(d_model)
        # self.feat_norm = nn.LayerNorm(d_model)
        self.use_pos_embed = use_pos_embed

        # self.query = nn.Parameter(torch.randn(num_queries, d_model))

        if use_pos_embed:
            self.query_pos_embed = nn.Embedding(num_queries, d_model)
            self.feat_pos_embed = nn.Embedding(num_feats, d_model)
        
    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        return self.attn(query, key, value, need_weights=True, attn_mask=attn_mask_)[0]

    def build_attention_mask(self, query_mask, feat_mask):

        extended_query_mask = query_mask.unsqueeze(-1).expand(-1, -1, feat_mask.shape[1])
        extended_feat_mask = feat_mask.unsqueeze(1).expand(-1, query_mask.shape[1], -1)
        extended_mask = extended_query_mask * extended_feat_mask

        extended_mask = extended_mask.to(dtype=query_mask.dtype)
        extended_mask = (1.0 - extended_mask) * -1000000.0

        return extended_mask

    def add_pos_embed(self, x, pos_embedding):
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        x_pos_embed = pos_embedding(position_ids)
        x = x + x_pos_embed
        return x

    def forward(self, query, feat, query_mask=None, feat_mask=None):
        
        # query = self.query + torch.zeros(feat.shape[0], self.query.shape[0], feat.shape[-1], dtype=feat.dtype, device=feat.device)

        if self.use_pos_embed:
            query = self.add_pos_embed(query, self.query_pos_embed)
            feat = self.add_pos_embed(feat, self.feat_pos_embed)

        # print(f"query: {query.shape}")
        # print(f"feat: {feat.shape}")
        
        if query_mask is None:
            query_mask = torch.ones((query.shape[0], query.shape[1]), dtype=feat.dtype, device=feat.device)
        
        if feat_mask is None:
            feat_mask = torch.ones((feat.shape[0], feat.shape[1]), dtype=feat.dtype, device=feat.device)

        # print(f"query_mask: {query_mask.shape}")
        # print(f"feat_mask: {feat_mask.shape}")

        attn_mask = self.build_attention_mask(query_mask, feat_mask)

        # print(f"attn_mask: {attn_mask.shape}")

        query = query.permute(1, 0, 2)
        feat = feat.permute(1, 0, 2)
        # query = self.attention(self.query_norm(query), self.feat_norm(feat), self.feat_norm(feat), attn_mask)
        query = self.attention(query, feat, feat, attn_mask)
        query = query.permute(1, 0, 2)

        return query

class UnifiedResampler(nn.Module):
    def __init__(self,
        q_dim, kv_dim
    ):
        super().__init__()

        self.query_projector = nn.Sequential(nn.LayerNorm(q_dim), 
                                             nn.Linear(q_dim, q_dim))
        
        self.key_projector = nn.Sequential(nn.LayerNorm(kv_dim), 
                                           nn.Linear(kv_dim, q_dim))
        
        self.value_projector = nn.Sequential(nn.LayerNorm(kv_dim), 
                                             nn.Linear(kv_dim, q_dim))

    def forward(self, x_q, x_kv):

        if len(x_kv.shape) == 3:
            hw = int(x_kv.size(1) ** 0.5)
            x_kv = rearrange(x_kv, "b (h w) d -> b d h w", h=hw, w=hw)


        # patchwise with square images
        patch_num = int(x_q.shape[1]**0.5)
        patch_size = x_kv.shape[-1] // patch_num
        # within patch attention
        x_kv = x_kv.permute(0,2,3,1)
        x_kv = x_kv.reshape(len(x_kv), patch_num, patch_size, patch_num, patch_size, x_kv.shape[-1])
        x_kv = x_kv.permute(0,1,3,2,4,5)
        x_kv = x_kv.reshape(len(x_kv), patch_num**2, patch_size**2, x_kv.shape[-1]).contiguous()

        # print(f"x_q: {x_q.shape}")
        # print(f"x_kv: {x_kv.shape}")

        # token attention
        embed_query = self.query_projector(x_q)
        embed_aux = self.key_projector(x_kv)
        embed_value = self.value_projector(x_kv) 
        embed_att = embed_query[:,:,None] @ (embed_aux.transpose(-1,-2) / (embed_aux.shape[-1]**0.5))
        embed_att = embed_att.nan_to_num()
        embed_feat = (embed_att.softmax(-1) @ embed_value).mean(2)
        
        return x_q, embed_feat


class GlobalCrossAttn(nn.Module):
    def __init__(self, q_dim, kv_dim, num_queries, num_feats):
        super().__init__()

        # self.num_feats = num_feats
        num_heads = q_dim // 128
        self.ca = CrossAttention(q_dim, num_heads, use_pos_embed=False)

        self.query_projector = nn.Sequential(nn.LayerNorm(q_dim), 
                                             nn.Linear(q_dim, q_dim))
        
        self.key_projector = nn.Sequential(nn.LayerNorm(kv_dim), 
                                           nn.Linear(kv_dim, q_dim))

    def divide_feature(self, x, num_tokens, scale_factor):

        hw = int(num_tokens ** 0.5)
        py = px = hw // scale_factor

        x = rearrange(x, 'b (h w) c -> h w b c', h=hw, w=hw)
        x = rearrange(x, '(py yy) (px xx) b c -> (py px b) (yy xx) c',
                      py=py, yy=scale_factor,
                      px=px, xx=scale_factor,
        )
        
        return x

    # def divide_feature(self, x_q, x_kv):
    #     patch_num = int(x_q.shape[1]**0.5)
    #     patch_size = x_kv.shape[-1] // patch_num
    #     x_kv = x_kv.permute(0,2,3,1)
    #     # print(f"x_kv: {x_kv.shape}")
    #     x_kv = x_kv.reshape(len(x_kv), patch_num, patch_size, patch_num, patch_size, x_kv.shape[-1])
    #     # print(f"x_kv: {x_kv.shape}")
    #     x_kv = x_kv.permute(0,1,3,2,4,5)
    #     # print(f"x_kv: {x_kv.shape}")
    #     x_kv = x_kv.reshape(len(x_kv), patch_num**2, patch_size**2, x_kv.shape[-1]).contiguous()
    #     # print(f"x_kv: {x_kv.shape}")

    #     x_q = x_q.view(-1, 1, x_q.shape[-1])
    #     # print(f"x_q: {x_q.shape}")
    #     x_kv = x_kv.view(-1, patch_size**2, x_kv.shape[-1])
    #     # print(f"x_kv: {x_kv.shape}")

    #     return x_q, x_kv

    def forward(self, x_q, x_kv):

        x_q = self.query_projector(x_q)
        x_kv = self.key_projector(x_kv)

        rearrange_x_q = self.divide_feature(x_q, x_q.shape[1], 1) 
        # print(f"rearrange_x_q: {rearrange_x_q.shape}")
        # rearrange_x_kv = self.divide_feature(x_kv, x_kv.shape[1], int(self.num_feats ** 0.5)) 
        rearrange_x_kv = self.divide_feature(x_kv, x_kv.shape[1], 1) 
        # print(f"rearrange_x_kv: {rearrange_x_kv.shape}")
        x_attn = self.ca(rearrange_x_q, rearrange_x_kv)
        x_attn = x_attn.view(x_q.shape[0], x_q.shape[1], x_q.shape[2])
        # print(f"x_attn: {x_attn.shape}")

        return x_attn

        # x_q = x_q + x_attn

        # return x_q
        

class CrossAttnWithUnifidResampler(nn.Module):
    def __init__(self,
        visual_hidden_size, hidden_size, hw,
    ):
        super().__init__()

        # self.ca = UnifiedResampler(hidden_size, visual_hidden_size)
        self.ca = UnifiedResampler(hidden_size, hidden_size)
        # self.sampler = nn.AdaptiveAvgPool2d((hw, hw))

    def forward(self, x, x_aux):
        
        if len(x_aux.shape) == 3:
            hw = int(x_aux.size(1) ** 0.5)
            x_aux = rearrange(x_aux, "b (h w) d -> b d h w", h=hw, w=hw)

        x_ca = self.ca(x, x_aux)

        # x_aux_sampler = self.sampler(x_aux)
        # x_aux_sampler = rearrange(x_aux_sampler, "b d h w -> b (h w) d")

        # x_out = torch.cat([x, x_ca], dim=-1)
        x_out = x + x_ca

        return x_out
        

class PAFPNHB1(nn.Module):
    def __init__(self, 
        visual_hidden_size, hidden_size, reg_depth, hw,
    ):
        super().__init__()

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        self.s1 = RegBlock(
            reg_depth,
            visual_hidden_size,
            hidden_size,
        )
        # self.sampler1 = nn.AdaptiveAvgPool2d((hw, hw))
        self.sampler1 = ConvNormAct(
            hidden_size, hidden_size, kernel_size=3, stride=2, padding=1, dilation=1, act_layer=nn.SiLU, norm_layer=LayerNorm2d,
        )
        self.s2 = RegBlock(
            reg_depth,
            hidden_size,
            hidden_size,
        )

        self.trans_layer1 = ConvNormAct(
            hidden_size, hidden_size, stride=1, dilation=1, act_layer=nn.SiLU, norm_layer=LayerNorm2d,
        )  

    def forward(self, x_inputs):

        reshaped_x_inputs = []
        for x in x_inputs:
            if len(x.shape) == 3:
                hw = int(x.size(1) ** 0.5)
                x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
            reshaped_x_inputs.append(x)

        x_s1, x_s2 = reshaped_x_inputs

        # print(f"x_s1: {x_s1.shape}")
        # print(f"x_s2: {x_s2.shape}")

        # x_s1_aug = x_s1 + F.interpolate(x_s2, scale_factor=2, mode='bilinear', align_corners=False)
        x_s1_aug = x_s1 + F.upsample(x_s2, scale_factor=2, mode='nearest')
        # print(f"x_s1_aug: {x_s1_aug.shape}")
        x_s1_new = self.s1(x_s1_aug)
        # print(f"x_s1_new: {x_s1_new.shape}")
        x_sampler1_new = self.sampler1(x_s1_new) + x_s2
        # print(f"x_sampler1_new: {x_sampler1_new.shape}")
        x_s2_new = self.s2(x_sampler1_new)
        # print(f"x_s2_new: {x_s2_new.shape}")

        x_out = rearrange(x_s2_new, "b d h w -> b (h w) d")
        # print(f"x_out: {x_out.shape}")

        return x_out

class PAFPNHB2(nn.Module):
    def __init__(self, 
        visual_hidden_size, hidden_size, reg_depth, hws,
    ):
        super().__init__()

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        self.s1 = RegBlock(
            reg_depth,
            visual_hidden_size,
            hidden_size,
        )
        # self.sampler1 = nn.AdaptiveAvgPool2d((hws[0], hws[0]))
        self.sampler1 = ConvNormAct(
            hidden_size, hidden_size, kernel_size=3, stride=2, padding=1, dilation=1, act_layer=nn.SiLU, norm_layer=LayerNorm2d,
        )
        self.s2 = RegBlock(
            reg_depth,
            hidden_size,
            hidden_size,
        )
        # self.sampler2 = nn.AdaptiveAvgPool2d((hws[1], hws[1]))
        self.sampler2 = ConvNormAct(
            hidden_size, hidden_size, kernel_size=3, stride=2, padding=1, dilation=1, act_layer=nn.SiLU, norm_layer=LayerNorm2d,
        )
        self.s3 = RegBlock(
            reg_depth,
            hidden_size,
            hidden_size,
        )

        self.trans_layer1 = ConvNormAct(
            hidden_size, hidden_size, stride=1, dilation=1, act_layer=nn.SiLU, norm_layer=LayerNorm2d,
        )

        self.trans_layer2 = ConvNormAct(
            hidden_size, hidden_size, stride=1, dilation=1, act_layer=nn.SiLU, norm_layer=LayerNorm2d,
        )     

    def forward(self, x_inputs):
        
        reshaped_x_inputs = []
        for x in x_inputs:
            if len(x.shape) == 3:
                hw = int(x.size(1) ** 0.5)
                x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
            reshaped_x_inputs.append(x)

        x_s1, x_s2, x_s3 = reshaped_x_inputs

        # print(f"x_s1: {x_s1.shape}") # [bs, 1024, 32, 32]
        # print(f"x_s2: {x_s2.shape}") # [bs, 1024, 16, 16]
        # print(f"x_s3: {x_s3.shape}") # [bs, 1024, 8, 8]

        # x_s2_aug = x_s2 + F.interpolate(x_s3, scale_factor=2, mode='bilinear', align_corners=False)
        # x_s1_aug = x_s1 + F.interpolate(x_s2_aug, scale_factor=2, mode='bilinear', align_corners=False)

        x_s2_aug = x_s2 + F.upsample(x_s3, scale_factor=2, mode='nearest')
        x_s1_aug = x_s1 + F.upsample(x_s2_aug, scale_factor=2, mode='nearest')

        x_s1_new = self.s1(x_s1_aug)
        x_sampler1_new = self.sampler1(x_s1_new) + self.trans_layer1(x_s2_aug)
        x_s2_new = self.s2(x_sampler1_new)
        x_sampler2_new = self.sampler2(x_s2_new) + x_s3
        x_s3_new = self.s3(x_sampler2_new)

        x_out = rearrange(x_s3_new, "b d h w -> b (h w) d")

        return x_out
    

class DenseHB1ConvNet(nn.Module):
    def __init__(self,
        visual_hidden_size, hidden_size, reg_depth, hw,
    ):
        super().__init__()

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        self.s1 = RegBlock(
            reg_depth,
            visual_hidden_size,
            hidden_size,
        )
        self.sampler1 = nn.AdaptiveAvgPool2d((hw, hw))
        self.s2 = RegBlock(
            reg_depth,
            # visual_hidden_size + hidden_size,
            visual_hidden_size + hidden_size + hidden_size,
            hidden_size,
        )

        # self.net = nn.Sequential(s1, sampler1, s2)

        self.ca = UnifiedResampler(hidden_size, visual_hidden_size)

        self.sampler_shortcut = nn.AdaptiveAvgPool2d((hw, hw))

        # self.trans_layer = nn.Sequential(
        #     ConvNormAct(
        #         visual_hidden_size, hidden_size, stride=1, dilation=1, act_layer=nn.SiLU, norm_layer=LayerNorm2d,
        #     ),
        #     nn.AvgPool2d(2, 2)
        # )

        # self.net = nn.Sequential(s1, sampler1, s2)


        # self.s1 = RegStage(
        #     depth=reg_depth,
        #     in_chs=visual_hidden_size,
        #     out_chs=hidden_size,
        #     stride=2, 
        #     act_layer=nn.SiLU,
        #     dilation=1,
        #     norm_layer=LayerNorm2d,
        # )

    def forward(self, x):
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)

        
        x_s1 = self.s1(x) # x_s1 = self.net[0](x) 
        # x_sampler1 = self.sampler1(x_s1) + self.trans_layer(x)

        x_sampler1_part1 = self.sampler1(x_s1)
        x_sampler1_ca = self.ca(rearrange(x_sampler1_part1, "b d h w -> b (h w) d"), x)
        hw = int(x_sampler1_ca.size(1) ** 0.5)
        x_sampler1_ca = rearrange(x_sampler1_ca, "b (h w) d -> b d h w", h=hw, w=hw)
        x_sampler1_shortcut = self.sampler_shortcut(x)

        x_sampler1 = torch.cat([x_sampler1_part1, x_sampler1_ca, x_sampler1_shortcut], dim=1)

        # x_sampler1 = torch.cat([self.sampler1(x_s1), self.sampler_shortcut(x)], dim=1) # x_sampler1 = torch.cat([self.net[1](x_s1), self.sampler_shortcut(x)], dim=1) 
        # print(f"x_sampler1: {x_sampler1.shape}")

        x_s2 = self.s2(x_sampler1) # x_s2 = self.net[2](x_sampler1)

        x_out = rearrange(x_s2, "b d h w -> b (h w) d")

        return x_out, x

class DenseHB2ConvNet(nn.Module):
    def __init__(self, visual_hidden_size, hidden_size, reg_depth, hw):
        super().__init__()

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )
    
        self.sampler2 = nn.AdaptiveAvgPool2d((hw, hw))
        self.s3 = RegBlock(
            reg_depth,
            # visual_hidden_size + hidden_size,
            visual_hidden_size + hidden_size + hidden_size,
            hidden_size,
        )

        self.ca = UnifiedResampler(hidden_size, visual_hidden_size)

        self.sampler_shortcut = nn.AdaptiveAvgPool2d((hw, hw))

        # self.trans_layer = nn.Sequential(
        #     ConvNormAct(
        #         hidden_size + hidden_size, hidden_size, stride=1, dilation=1, act_layer=nn.SiLU, norm_layer=LayerNorm2d,
        #     ),
        #     nn.AvgPool2d(2, 2)
        # )

        # self.net = nn.Sequential(sampler2, s3)

        # self.s1 = RegStage(
        #     depth=reg_depth,
        #     in_chs=visual_hidden_size,
        #     out_chs=hidden_size,
        #     stride=2, 
        #     act_layer=nn.SiLU,
        #     dilation=1,
        #     norm_layer=LayerNorm2d,
        # )

        # self.s2 = RegStage(
        #     depth=reg_depth,
        #     in_chs=hidden_size,
        #     out_chs=hidden_size * 2,
        #     stride=2, 
        #     act_layer=nn.SiLU,
        #     dilation=1,
        #     norm_layer=LayerNorm2d,
        # )

    def forward(self, x, x_pre=None):
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)

        hw = int(x_pre.size(1) ** 0.5)
        x_pre = rearrange(x_pre, "b (h w) d -> b d h w", h=hw, w=hw)

        # x_sampler2 = self.sampler2(x) + self.trans_layer(x_pre)

        x_sampler2_part1 = self.sampler2(x)
        x_sampler2_ca = self.ca(rearrange(x_sampler2_part1, "b d h w -> b (h w) d"), x_pre)
        hw = int(x_sampler2_ca.size(1) ** 0.5)
        x_sampler2_ca = rearrange(x_sampler2_ca, "b (h w) d -> b d h w", h=hw, w=hw)
        x_sampler2_shortcut = self.sampler_shortcut(x_pre)

        x_sampler2 = torch.cat([x_sampler2_part1, x_sampler2_ca, x_sampler2_shortcut], dim=1)

        # x_sampler2 = torch.cat([self.sampler2(x), self.sampler_shortcut(x_pre)], dim=1)
        # print(f"x_sampler2: {x_sampler2.shape}")
        x_s3 = self.s3(x_sampler2)

        x_out = rearrange(x_s3, "b d h w -> b (h w) d")

        # x_s2 = self.s2(x)
        # # x_s2 = self.s2(self.s1(x))
        # x_out = rearrange(x_s2, "b d h w -> b (h w) d")

        return x_out

class DenseMLP(nn.Module):
    def __init__(self, visual_hidden_size, hidden_size, mlp_depth, mlp_act_func):
        super().__init__()

        self.compress_proj1 = build_mlp(mlp_depth, visual_hidden_size * 4, hidden_size, mlp_act_func)
        self.compress_proj2 = build_mlp(mlp_depth, hidden_size * 4, hidden_size, mlp_act_func)

    def pixel_shuffle(self, x, scale_factor=0.5, image_size=None):
        n, wh, c = x.shape
        if image_size is not None:
            w, h = image_size
        else:
            w = h = int(wh ** 0.5)
            
        x = x.view(n, w, h, c)
        # print(f"x: {x.shape}")

        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
    
        x = x.view(n, -1, x.shape[-1])
        # print(f"x: {x.shape}")

        return x

    def forward(self, x_thumbnail, x_tiles):
        x_thumbnail = self.pixel_shuffle(x_thumbnail)
        # print(f"x_thumbnail: {x_thumbnail.shape}")
        x_thumbnail = self.compress_proj1(x_thumbnail)
        # print(f"x_thumbnail: {x_thumbnail.shape}")
        x_tiles = self.pixel_shuffle(x_tiles)
        # print(f"x_tiles: {x_tiles.shape}")
        x_tiles = self.compress_proj1(x_tiles)
        # print(f"x_tiles: {x_tiles.shape}")
        x_tiles = self.pixel_shuffle(x_tiles)
        # print(f"x_tiles: {x_tiles.shape}")
        x_tiles = self.compress_proj2(x_tiles)
        # print(f"x_tiles: {x_tiles.shape}")

        return x_thumbnail, x_tiles

class HB1ConvNet(nn.Module):
    def __init__(self,
        visual_hidden_size, hidden_size, reg_depth, hws,
    ):
        super().__init__()

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        # s1 = RegBlock(
        #     reg_depth,
        #     visual_hidden_size,
        #     hidden_size,
        # )
        # sampler1 = nn.AdaptiveAvgPool2d((hw, hw))
        # s2 = RegBlock(
        #     reg_depth,
        #     hidden_size,
        #     hidden_size,
        # )
        # self.net = nn.Sequential(s1, sampler1, s2)
        
        net = [RegBlock(reg_depth, visual_hidden_size, hidden_size)]
        for hw in hws:
            net.append(nn.AdaptiveAvgPool2d((hw, hw)))
            net.append(RegBlock(reg_depth, hidden_size, hidden_size))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)

        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")

        return x, x

        # x_s1 = self.net[0](x)
        # x_sampler1 = self.net[1](x_s1)
        # x_s2 = self.net[2](x_sampler1)
        # x_s2 = rearrange(x_s2, "b d h w -> b (h w) d")

        # return x_s2, x_s1

class HB2ConvNet(nn.Module):
    def __init__(self, visual_hidden_size, hidden_size, reg_depth, hws):
        super().__init__()

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )
    
        # sampler2 = nn.AdaptiveAvgPool2d((hw, hw))

        # s3 = RegBlock(
        #     reg_depth,
        #     hidden_size,
        #     hidden_size,
        # )
        # self.net = nn.Sequential(sampler2, s3)

        net = []
        for hw in hws:
            net.append(nn.AdaptiveAvgPool2d((hw, hw)))
            net.append(RegBlock(reg_depth, hidden_size, hidden_size))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)

        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        
        return x

        # x_sampler2 = self.net[0](x)
        # x_s3 = self.net[1](x_sampler2)
        # x_s3 = rearrange(x_s3, "b d h w -> b (h w) d")

        # return x_s3


class FusionProjectorModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = FusionProjectorConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: FusionProjectorConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False

        # self.model = FusionProjector(config)

        visual_hidden_size = config.visual_hidden_size
        llm_hidden_size = config.llm_hidden_size
        hidden_size = config.hidden_size
        hws = [int(hw ** 0.5) for hw in config.hws]
        vit_num_tokens = config.vit_num_tokens
        num_heads = config.num_heads
        reg_depth = config.reg_depth
        mlp_depth = config.mlp_depth
        mlp_act_func = config.mlp_act_func

        projector_type = config.projector_type
        self.projector_type = projector_type

        self.channel_fusion = config.channel_fusion
        channel_fusion_hs1 = config.channel_fusion_hs1
        channel_fusion_hs2 = config.channel_fusion_hs2

        self.pos_emb = build_hb_pos_embeds(vit_num_tokens, visual_hidden_size)

        if projector_type == 'densenet':
            self.hb1 = DenseHB1ConvNet(visual_hidden_size, hidden_size, reg_depth, hws[0])
            self.hb2 = DenseHB2ConvNet(visual_hidden_size, hidden_size, reg_depth, hws[1])

            # self.trans_layer = RegStage(
            #     depth=1,
            #     in_chs=hidden_size,
            #     out_chs=hidden_size * 2,
            #     stride=1, 
            #     act_layer=nn.SiLU,
            #     dilation=1,
            #     norm_layer=LayerNorm2d,
            # )
            # self.hb = DenseConvNet(visual_hidden_size, hidden_size, reg_depth, hws)
        elif projector_type == "pixel_shuffle":
            self.dense_mlp = DenseMLP(visual_hidden_size, hidden_size, mlp_depth, mlp_act_func)
        else:
            self.hb1 = HB1ConvNet(visual_hidden_size, hidden_size, reg_depth, [hws[0]])
            self.hb2 = HB2ConvNet(visual_hidden_size, hidden_size, reg_depth, [hws[1]])

            # self.hb1 = HB1ConvNet(visual_hidden_size, hidden_size, reg_depth, hws[:2])
            # self.hb2 = HB2ConvNet(visual_hidden_size, hidden_size, reg_depth, hws[2:])


        # self.ca = GlobalCrossAttn(1152, 1024, 1, 1)

        if self.channel_fusion:
            self.ca = UnifiedResampler(channel_fusion_hs1, channel_fusion_hs2)

        # if projector_type == "tokenpacker":
        #     self.ca_thumbnail = GlobalCrossAttn(hidden_size, num_heads, 1, 4)
        #     self.ca_tiles = GlobalCrossAttn(hidden_size, num_heads, 1, 16)

        #     # self.globalca = GlobalCrossAttn(hidden_size, num_heads, 
        #     #                                 config.hws[0], config.hws[1], 
        #     #                                 2, 4)
            
        #     # self.ca_thumbnail = CrossAttnWithUnifidResampler(visual_hidden_size, hidden_size, hws[0])
        #     # self.ca_tiles = CrossAttnWithUnifidResampler(visual_hidden_size, hidden_size, hws[1])

        # if projector_type == 'pafpn':
        #     self.fpn1 = PAFPNHB1(hidden_size, hidden_size, reg_depth, hws[0])
        #     self.fpn2 = PAFPNHB2(hidden_size, hidden_size, reg_depth, hws)

        # self.readout = nn.Sequential(
        #     build_mlp(mlp_depth, hidden_size, visual_hidden_size, mlp_act_func),
        #     build_mlp(mlp_depth, visual_hidden_size, llm_hidden_size, mlp_act_func)
        # )

        # self.readout = build_mlp(mlp_depth, hidden_size * 2, llm_hidden_size, mlp_act_func)

        # self.ca = CrossAttention(hidden_size, num_heads, use_pos_embed=False)

        self.readout = build_mlp(mlp_depth, hidden_size, llm_hidden_size, mlp_act_func)

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            if isinstance(output, tuple):
                for o in output:
                    o.requires_grad_(True)
            else:
                output.requires_grad_(True)

        # self.model.register_forward_hook(make_inputs_require_grad)

        if self.projector_type == "pixel_shuffle":
            self.dense_mlp.register_forward_hook(make_inputs_require_grad)
        else:
            self.hb1.register_forward_hook(make_inputs_require_grad)
            self.hb2.register_forward_hook(make_inputs_require_grad)

        if self.channel_fusion:
            self.ca.register_forward_hook(make_inputs_require_grad)

        # self.hb.register_forward_hook(make_inputs_require_grad)

        # if self.projector_type == 'densenet':
        #     self.trans_layer.register_forward_hook(make_inputs_require_grad)

        # if self.projector_type == 'pafpn':
        #     self.fpn1.register_forward_hook(make_inputs_require_grad)
        #     self.fpn2.register_forward_hook(make_inputs_require_grad)

        # if self.projector_type == 'tokenpacker':
        #     # self.globalca.register_forward_hook(make_inputs_require_grad)
        #     self.ca_thumbnail.register_forward_hook(make_inputs_require_grad)
        #     self.ca_tiles.register_forward_hook(make_inputs_require_grad)

        self.readout.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FusionProjectorModel):
            module.gradient_checkpointing = value

    def vote_and_mix(self, x, r=0.8):
        bs, n, _ = x.shape

        x_norm = x / x.norm(dim=-1, keepdim=True)
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))

        for i in range(sim.size(0)):
            sim[i].fill_diagonal_(float('-inf'))

        v_w, v_i = sim.max(1)

        score = torch.zeros((bs, n), device=x.device, dtype=x.dtype).scatter_add_(-1, v_i, v_w)
        r_id = score.argsort(-1)[:, :int(n * (1 - r))]
        p_id = score.argsort(-1)[:,int(n * (1 - r)):]

        w = []

        for i in range(bs):
            w.append(torch.index_select(torch.index_select(sim[i], dim=0, index=p_id[i]), dim=1, index=r_id[i]))

        w = torch.stack(w).to(x.device)
        w = F.softmax(w, dim=-1)

        x_p, x_r = [], []

        for i in range(bs):
            x_p.append(torch.index_select(x[i], dim=0, index=p_id[i]))
            x_r.append(torch.index_select(x[i], dim=0, index=r_id[i]))
        x_p = torch.stack(x_p).to(x.device)
        x_r = torch.stack(x_r).to(x.device)

        x_mix = torch.bmm(w.transpose(1, 2), x_p)

        x_out = x_r + x_mix

        return x_out


    def vit_channel_fusion(self, x_input):

        x, x_aux = x_input

        # print(f"x: {x.shape}, x_aux: {x_aux.shape}")

        if self.gradient_checkpointing and self.training:
            x, x_attn = torch.utils.checkpoint.checkpoint(self.ca, x, x_aux)
        else:
            x, x_attn = self.ca(x, x_aux)

        # print(f"x: {x.shape}, x_attn: {x_attn.shape}")

        x_out = torch.cat([x, x_attn, x_aux], dim=-1)

        # print(f"x_out: {x_out.shape}")
        # quit()
        
        return x_out

    def forward(self, x_input):

        # print(f"self.gradient_checkpointing and self.training: {self.gradient_checkpointing and self.training}")

        if self.gradient_checkpointing and self.training:
            # layer_outputs = torch.utils.checkpoint.checkpoint(self.model, x)

            if isinstance(x_input, list):
                x_thumbnail, x_tiles = x_input

                if self.projector_type == "pixel_shuffle":
                    layer_outputs_thumbnail, layer_outputs_tiles = torch.utils.checkpoint.checkpoint(self.dense_mlp, x_thumbnail, x_tiles)
                else:

                    x_thumbnail = x_thumbnail + self.pos_emb
                    x_tiles = x_tiles + self.pos_emb
                    
                    # layer_outputs_thumbnail = torch.utils.checkpoint.checkpoint(self.hb, x_thumbnail, "x_s2", use_reentrant=False)
                    layer_outputs_thumbnail, hidden_state_tumbnail = torch.utils.checkpoint.checkpoint(self.hb1, x_thumbnail, use_reentrant=False)
                    # layer_outputs_thumbnail, hidden_state_tumbnail = torch.utils.checkpoint.checkpoint(self.hb1, x_thumbnail)
                    # print(f"layer_outputs_thumbnail: {layer_outputs_thumbnail.shape}")

                    # layer_outputs_tiles = torch.utils.checkpoint.checkpoint(self.hb, x_tiles, "x_s3", use_reentrant=False)
                    # layer_outputs_tiles, hidden_state_tiles = torch.utils.checkpoint.checkpoint(self.hb1, x_tiles, use_reentrant=False)
                    layer_outputs_tiles, hidden_state_tiles = torch.utils.checkpoint.checkpoint(self.hb1, x_tiles, use_reentrant=False)
                    # if self.projector_type == 'pafpn':
                    #     tiles_fpn_inputs = [hidden_state_tiles, layer_outputs_tiles]
                    # # print(f"layer_outputs_tiles: {layer_outputs_tiles.shape}")

                    # layer_outputs_tiles = torch.utils.checkpoint.checkpoint(self.hb2, layer_outputs_tiles)
                    # layer_outputs_tiles = torch.utils.checkpoint.checkpoint(self.hb2, x_tiles)

                    if self.projector_type == 'densenet':
                        layer_outputs_tiles = torch.utils.checkpoint.checkpoint(self.hb2, layer_outputs_tiles, x_tiles)
                    else:
                        layer_outputs_tiles = torch.utils.checkpoint.checkpoint(self.hb2, layer_outputs_tiles)
                        # layer_outputs_tiles = torch.utils.checkpoint.checkpoint(self.hb2, layer_outputs_tiles_tmp)

                    # if self.projector_type == 'pafpn':
                    #     tiles_fpn_inputs.append(layer_outputs_tiles)
                    # print(f"layer_outputs_tiles: {layer_outputs_tiles.shape}")

                    # if self.projector_type == "densenet":
                    #     hw = int(layer_outputs_thumbnail.size(1) ** 0.5)
                    #     layer_outputs_thumbnail = rearrange(layer_outputs_thumbnail, "b (h w) d -> b d h w", h=hw, w=hw)
                    #     layer_outputs_thumbnail = torch.utils.checkpoint.checkpoint(self.trans_layer, layer_outputs_thumbnail)
                    #     layer_outputs_thumbnail = rearrange(layer_outputs_thumbnail, "b d h w -> b (h w) d")

                    # if self.projector_type == 'tokenpacker':
                    #     hidden_state_tumbnail = rearrange(hidden_state_tumbnail, "b d h w -> b (h w) d")
                    #     hidden_state_tiles = rearrange(hidden_state_tiles, "b d h w -> b (h w) d")

                    #     # layer_outputs_thumbnail, layer_outputs_tiles = torch.utils.checkpoint.checkpoint(self.globalca, layer_outputs_thumbnail, layer_outputs_tiles, hidden_state_tumbnail, hidden_state_tiles)
                    #     # layer_outputs_tiles = torch.utils.checkpoint.checkpoint(self.globalca, layer_outputs_tiles, hidden_state_tiles)

                    #     layer_outputs_thumbnail = torch.utils.checkpoint.checkpoint(self.ca_thumbnail, layer_outputs_thumbnail, hidden_state_tumbnail)
                    #     layer_outputs_tiles = torch.utils.checkpoint.checkpoint(self.ca_tiles, layer_outputs_tiles, hidden_state_tiles)
                    #     # layer_outputs_tiles = torch.utils.checkpoint.checkpoint(self.ca_tiles, layer_outputs_tiles, layer_outputs_tiles_tmp)

                    # if self.projector_type == 'pafpn':
                    #     layer_outputs_thumbnail = torch.utils.checkpoint.checkpoint(self.fpn1, [hidden_state_tumbnail, layer_outputs_thumbnail])
                    #     layer_outputs_tiles = torch.utils.checkpoint.checkpoint(self.fpn2, tiles_fpn_inputs)

                bs = layer_outputs_thumbnail.shape[0]
                layer_outputs_tiles = layer_outputs_tiles.contiguous().view(bs, -1, layer_outputs_tiles.shape[2])
                # print(f"reshapeed layer_outputs_tiles: {layer_outputs_tiles.shape}")

                # if self.projector_type == 'vote_and_mix':
                #     layer_outputs_tiles_reduced = self.vote_and_mix(layer_outputs_tiles)
                #     layer_outputs_tiles = torch.utils.checkpoint.checkpoint(self.ca, layer_outputs_tiles_reduced, layer_outputs_tiles)
                #     # print(f"vote_and_mix layer_outputs_tiles: {layer_outputs_tiles.shape}")

                layer_outputs = torch.cat([layer_outputs_thumbnail, layer_outputs_tiles], dim=1)
                # print(f"layer_outputs: {layer_outputs.shape}")

                layer_outputs = torch.utils.checkpoint.checkpoint(self.readout, layer_outputs)
                # print(f"layer_outputs: {layer_outputs.shape}")
            else:
                x_thumbnail = x_input + self.pos_emb
                layer_outputs_thumbnail, hidden_state_tumbnail = torch.utils.checkpoint.checkpoint(self.hb1, x_thumbnail)
                layer_outputs = torch.utils.checkpoint.checkpoint(self.readout, layer_outputs_thumbnail)
        else:
            # layer_outputs = self.model(x)

            if isinstance(x_input, list):
                print('[x_thumbnail, x_tiles] as x_input to FusionProjectorModel forward')
                x_thumbnail, x_tiles = x_input

                if self.projector_type == "pixel_shuffle":
                    layer_outputs_thumbnail, layer_outputs_tiles = self.dense_mlp(x_thumbnail, x_tiles)
                else:
                    x_thumbnail = x_thumbnail + self.pos_emb
                    x_tiles = x_tiles + self.pos_emb

                    # layer_outputs_thumbnail = self.hb(x_thumbnail, return_x="x_s2")
                    layer_outputs_thumbnail, hidden_state_tumbnail = self.hb1(x_thumbnail)
                    # print(f"layer_outputs_thumbnail: {layer_outputs_thumbnail.shape}")
                    # print(f"hidden_state_tumbnail: {hidden_state_tumbnail.shape}")

                    # layer_outputs_tiles = self.hb(x_tiles, return_x="x_s3")
                    # layer_outputs_tiles, hidden_state_tiles = self.hb1(x_tiles)
                    layer_outputs_tiles, hidden_state_tiles = self.hb1(x_tiles)
                    # print(f"layer_outputs_tiles: {layer_outputs_tiles_tmp.shape}")
                    # print(f"hidden_state_tiles: {hidden_state_tiles.shape}")
                    
                    # layer_outputs_tiles = self.hb2(layer_outputs_tiles)
                    # layer_outputs_tiles = self.hb2(x_tiles)
                    # print(f"layer_outputs_tiles: {layer_outputs_tiles.shape}")
                    # quit()

                    # if self.projector_type == 'pafpn':
                    #     tiles_fpn_inputs = [hidden_state_tiles, layer_outputs_tiles]
                    if self.projector_type == 'densenet':
                        layer_outputs_tiles = self.hb2(layer_outputs_tiles, x_tiles)
                    else:
                        # layer_outputs_tiles = self.hb2(layer_outputs_tiles)
                        layer_outputs_tiles = self.hb2(layer_outputs_tiles)
                    # if self.projector_type == 'pafpn':
                    #     tiles_fpn_inputs.append(layer_outputs_tiles)
                    # print(f"layer_outputs_tiles: {layer_outputs_tiles.shape}")

                    # if self.projector_type == "densenet":
                    #     hw = int(layer_outputs_thumbnail.size(1) ** 0.5)
                    #     layer_outputs_thumbnail = rearrange(layer_outputs_thumbnail, "b (h w) d -> b d h w", h=hw, w=hw)
                    #     layer_outputs_thumbnail = self.trans_layer(layer_outputs_thumbnail)
                    #     layer_outputs_thumbnail = rearrange(layer_outputs_thumbnail, "b d h w -> b (h w) d")
                    #     # print(f"layer_outputs_thumbnail: {layer_outputs_thumbnail.shape}")

                    # if self.projector_type == 'tokenpacker':                
                    #     hidden_state_tumbnail = rearrange(hidden_state_tumbnail, "b d h w -> b (h w) d")
                    #     hidden_state_tiles = rearrange(hidden_state_tiles, "b d h w -> b (h w) d")

                    #     # layer_outputs_thumbnail, layer_outputs_tiles = self.globalca(layer_outputs_thumbnail, layer_outputs_tiles, hidden_state_tumbnail, hidden_state_tiles)
                    #     # layer_outputs_tiles = self.globalca(layer_outputs_tiles, hidden_state_tiles)

                    #     layer_outputs_thumbnail = self.ca_thumbnail(layer_outputs_thumbnail, hidden_state_tumbnail)
                    #     layer_outputs_tiles = self.ca_tiles(layer_outputs_tiles, hidden_state_tiles)
                    #     # layer_outputs_tiles = self.ca_tiles(layer_outputs_tiles, layer_outputs_tiles_tmp)

                    #     # print(f"layer_outputs_thumbnail: {layer_outputs_thumbnail.shape}")
                    #     # print(f"layer_outputs_tiles: {layer_outputs_tiles.shape}")
            
                    # if self.projector_type == 'pafpn':
                    #     layer_outputs_thumbnail = self.fpn1([hidden_state_tumbnail, layer_outputs_thumbnail])
                    #     layer_outputs_tiles = self.fpn2(tiles_fpn_inputs)

                bs = layer_outputs_thumbnail.shape[0]
                layer_outputs_tiles = layer_outputs_tiles.contiguous().view(bs, -1, layer_outputs_tiles.shape[2])
                # print(f"reshapeed layer_outputs_tiles: {layer_outputs_tiles.shape}")

                if self.projector_type == 'vote_and_mix':
                    layer_outputs_tiles = self.vote_and_mix(layer_outputs_tiles)
                    # layer_outputs_tiles = self.ca(layer_outputs_tiles_reduced, layer_outputs_tiles)
                    print(f"vote_and_mix layer_outputs_tiles: {layer_outputs_tiles.shape}")

                layer_outputs = torch.cat([layer_outputs_thumbnail, layer_outputs_tiles], dim=1)
                # print(f"layer_outputs: {layer_outputs.shape}")

                layer_outputs = self.readout(layer_outputs)
                # print(f"layer_outputs: {layer_outputs.shape}")
            else:
                print('x_thumbnail as x_input to FusionProjectorModel forward')
                x_thumbnail = x_input + self.pos_emb
                layer_outputs_thumbnail, hidden_state_tumbnail = self.hb1(x_thumbnail)
                layer_outputs = self.readout(layer_outputs_thumbnail)

        return layer_outputs
