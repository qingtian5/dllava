# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from .configuration_projector import ProjectorConfig


class ProjectorModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = ProjectorConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False

        self.do_pixel_shuffle = config.pixel_shuffle

        modules = [
            nn.Linear(
                config.visual_hidden_size,
                config.llm_hidden_size,
                bias=config.bias)
        ]
        for _ in range(1, config.depth):
            modules.append(ACT2FN[config.hidden_act])
            modules.append(
                nn.Linear(
                    config.llm_hidden_size,
                    config.llm_hidden_size,
                    bias=config.bias))
        self.model = nn.Sequential(*modules)

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.model.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ProjectorModel):
            module.gradient_checkpointing = value

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
        # quit()
        
        return x

    def forward(self, x):
        if self.do_pixel_shuffle:
            if isinstance(x, tuple):
                x = self.pixel_shuffle(x[0], image_size=x[1])
            else:
                x = self.pixel_shuffle(x)
            
        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(self.model, x)
        else:
            layer_outputs = self.model(x)
        return layer_outputs
