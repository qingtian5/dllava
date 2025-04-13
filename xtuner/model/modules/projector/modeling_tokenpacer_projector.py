# Copyright (c) OpenMMLab. All rights reserved.
import math
from functools import partial
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
# from .modeling_fusion_projector import CrossAttention

from .configuration_tokenpacker import TokenPackerConfig

class TokenPacker(nn.Module):
    def __init__(
            self,
            raw_grid=24,
            embed_dim=1024,
            num_heads=1024//128,
            kv_dim=1024,
            hidden_size=4096,
            scale_factor=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_func='gelu'
    ):
        super().__init__()
        # if raw_grid%scale_factor!=0:
        #     raise ValueError("scale_factor must be divisible by grid size")
        # self.raw_grid = raw_grid
        # self.grid_size = raw_grid//scale_factor
        # self.num_queries = self.grid_size ** 2
        self.num_queries = raw_grid
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.q_proj_1 = nn.Linear(kv_dim, embed_dim, bias=False)

        k_modules = [nn.Linear(kv_dim, embed_dim)]
        for _ in range(1,2):
            # k_modules.append(nn.GELU())
            k_modules.append(ACT2FN[act_func])
            k_modules.append(nn.Linear(embed_dim, embed_dim))
        self.k_proj_1 = nn.Sequential(*k_modules)

        v_modules = [nn.Linear(kv_dim, embed_dim)]
        for _ in range(1,2):
            # v_modules.append(nn.GELU())
            v_modules.append(ACT2FN[act_func])
            v_modules.append(nn.Linear(embed_dim, embed_dim))
        self.v_proj_1 = nn.Sequential(*v_modules)

        # kv_modules = [nn.Linear(kv_dim, embed_dim)]
        # for _ in range(1,2):
        #     kv_modules.append(nn.GELU())
        #     kv_modules.append(nn.Linear(embed_dim, embed_dim))
        # self.kv_proj_1 = nn.Sequential(*kv_modules)

        self.ln_q_1 = norm_layer(embed_dim)
        self.ln_k_1 = norm_layer(embed_dim)
        self.ln_v_1 = norm_layer(embed_dim)

        self.clip_attn = nn.MultiheadAttention(embed_dim, num_heads)

        # self.clip_attn = CrossAttention(embed_dim, num_heads, num_queries=self.num_queries, num_feats=raw_grid ** 2)

        # modules = [nn.Linear(embed_dim, hidden_size)]
        # for _ in range(1, 2):
        #     # modules.append(nn.GELU())
        #     modules.append(ACT2FN[act_func])
        #     modules.append(nn.Linear(hidden_size, hidden_size))
        # self.mlp = nn.Sequential(*modules)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def divide_feature(self, x, kernel_size, token_num, N, c):
        h = w = int(token_num**0.5)

        reshape_x = x.reshape(h, w, N, c).reshape(h//kernel_size, kernel_size, w, N, c)
        reshape_x = reshape_x.permute(0,2,1,3,4)
        reshape_x = reshape_x.reshape(h//kernel_size, w//kernel_size, kernel_size, kernel_size, N, c)
        reshape_x = reshape_x.permute(0,1,3,2,4,5).reshape(h//kernel_size, w//kernel_size, kernel_size*kernel_size, N, c)
        reshape_x = reshape_x.permute(2,0,1,3,4).reshape(kernel_size*kernel_size, -1, c)

        return reshape_x

    def forward(self, x, x_feat, attn_mask=None):

        # x_multi = x[1] # mulit-level
        # x = x[0] # original single-level

        # print(f"x_multi: {x_multi.shape}")
        # print(f"x: {x.shape}")

        # print(f"x: {x.shape}")
        # print(f"x_feat: {x_feat.shape}")
       

        # key = self.ln_k_1(self.k_proj_1(x_multi)).permute(1, 0, 2)
        key = self.ln_k_1(self.k_proj_1(x_feat)).permute(1, 0, 2)
        # print(f"key: {key.shape}")
        # value = self.ln_v_1(self.v_proj_1(x_multi)).permute(1, 0, 2)
        value = self.ln_v_1(self.v_proj_1(x_feat)).permute(1, 0, 2)
        # print(f"value: {value.shape}")

        # key = self.kv_proj_1(x_multi).permute(1, 0, 2)

        token_num, N, c = key.shape

        # q = F.interpolate(x.reshape(x.shape[0],self.raw_grid,self.raw_grid,-1).float().permute(0,3,1,2), size=(self.grid_size, self.grid_size), mode='bilinear').permute(0,2,3,1) ## fix
        # print(f"1 q: {q.shape}")
        # q = q.reshape(q.shape[0], -1, q.shape[-1]).to(x.dtype)
        # print(f"2 q: {q.shape}")

        query = self.ln_q_1(self.q_proj_1(x)).permute(1, 0, 2)
        # print(f"query: {query.shape}")

        # query = self.q_proj_1(q).permute(1, 0, 2)
        # print(f"query: {query.shape}")

        reshape_query = self.divide_feature(query, 1, self.num_queries, N, c)
        # print(f"reshape_query: {reshape_query.shape}")
        reshape_key = self.divide_feature(key, self.scale_factor, token_num, N, c)
        # print(f"reshape_key: {reshape_key.shape}")
        reshape_value = self.divide_feature(value, self.scale_factor, token_num, N, value.shape[-1])
        # print(f"reshape_value: {reshape_value.shape}")

        # print(f"attn_mask: {attn_mask}")
        out = self.clip_attn(
            reshape_query,
            reshape_key,
            reshape_value,
            attn_mask=attn_mask)[0]
        # print(f"out: {out.shape}")

        # out = self.clip_attn(
        #     reshape_query.permute(1, 0, 2),
        #     reshape_key.permute(1, 0, 2)
        # )
        # print(f"out: {out.shape}")

        # x_out = out
        out = out.reshape(self.num_queries, N, -1)
        out = out.permute(1, 0, 2)
        # print(f"out: {out.shape}")

        # x = self.mlp(x)
        # print(f"2 x: {x.shape}")
        # quit()

        return out 

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

class TokenPackerProjectorModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = TokenPackerConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: TokenPackerConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False

        self.model = TokenPacker(
            raw_grid=config.raw_grid,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            kv_dim=config.visual_hidden_size,
            hidden_size=config.llm_hidden_size,
            scale_factor=config.scale_factor,
        )

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.model.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, TokenPackerProjectorModel):
            module.gradient_checkpointing = value

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(self.model, x)
        else:
            layer_outputs = self.model(x)
        return layer_outputs
