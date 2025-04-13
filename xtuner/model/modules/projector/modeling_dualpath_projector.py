# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from .configuration_dualpath_projector import DualPathProjectorConfig
from .configuration_honeybee import HoneybeeProjectorConfig
# from .modeling_projector import ProjectorModel
from .modeling_honeybee_projector import HoneybeeProjectorModel
from .honeybee_projectors import CAbstractor, DAbstractor

class DualPathProjectorModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = DualPathProjectorConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: DualPathProjectorConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False

        self.config_thumbnail = HoneybeeProjectorConfig(**config.config_thumbnail)
        self.config_tiles = HoneybeeProjectorConfig(**config.config_tiles)

        # self.projector_thumbnail = ProjectorModel(self.config_thumbnail)
            
        self.projector_thumbnail = HoneybeeProjectorModel(self.config_thumbnail)
        
        self.projector_tiles = HoneybeeProjectorModel(self.config_tiles)
            

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.projector_thumbnail.register_forward_hook(make_inputs_require_grad)
        self.projector_tiles.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, DualPathProjectorModel):
            module.gradient_checkpointing = value

    def forward(self, x_tuple):
        x_thumbnail, x_tiles = x_tuple

        # print(f"x_thumbnail: {x_thumbnail.shape}")
        # print(f"x_tiles: {x_tiles.shape}")
            
        if self.gradient_checkpointing and self.training:
            layer_outputs_thumbnail = torch.utils.checkpoint.checkpoint(self.projector_thumbnail, x_thumbnail)
            layer_outputs_tiles = torch.utils.checkpoint.checkpoint(self.projector_tiles, x_tiles)
        else:
            layer_outputs_thumbnail = self.projector_thumbnail(x_thumbnail)
            layer_outputs_tiles = self.projector_tiles(x_tiles)

        # print(f"layer_outputs_thumbnail: {layer_outputs_thumbnail.shape}")
        # print(f"layer_outputs_tiles: {layer_outputs_tiles.shape}")

        bs = layer_outputs_thumbnail.shape[0]
        layer_outputs_tiles = layer_outputs_tiles.view(bs, -1, layer_outputs_tiles.shape [2])
        # print(f"reshapeed layer_outputs_tiles: {layer_outputs_tiles.shape}")

        layer_outputs = torch.cat([layer_outputs_thumbnail, layer_outputs_tiles], dim=1)
        # print(f"layer_outputs: {layer_outputs.shape}")

        return layer_outputs
