# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from .configuration_ldp import LDPProjectorConfig

class FeatureIRLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class TokenDownLayer(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.dwn = nn.Sequential(
            nn.AdaptiveAvgPool2d(shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwn(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class PosInjectLayer(nn.Module):
    # https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.peg = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=True, groups=out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        cnn_feat = x.transpose(1, 2).view(b, c, h, h)
        x = self.peg(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        return x

class LDPNetV2Projector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        inc, ouc = config.visual_hidden_size, config.llm_hidden_size
        dw_size = config.dw_size
        self.mlp = FeatureIRLayer(inc, ouc)
        self.dwn = TokenDownLayer(dw_size)
        self.peg = PosInjectLayer(ouc, ouc, stride=1)

    def forward(self, x):
        # print(f"1 x: {x.shape}")
        x = self.mlp(x)
        # print(f"2 x: {x.shape}")
        x = self.dwn(x)
        # print(f"3 x: {x.shape}")
        x = self.peg(x)
        # print(f"4 x: {x.shape}")
        return x


class LDPProjectorModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = LDPProjectorConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: LDPProjectorConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False

        self.model = LDPNetV2Projector(config)

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.model.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LDPProjectorModel):
            module.gradient_checkpointing = value

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(self.model, x)
        else:
            layer_outputs = self.model(x)
        return layer_outputs
