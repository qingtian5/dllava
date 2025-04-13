# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from .configuration_honeybee import HoneybeeProjectorConfig, HoneybeeVisualProjectorConfig
from .honeybee_projectors import CAbstractor, DAbstractor

class HoneybeeProjectorModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = HoneybeeProjectorConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: HoneybeeProjectorConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False

        """Build projector (abstractor) and query_tokens (optionally for resampler)"""
        self.proj_config = config.visual_projector_config
        self.proj_type = config.proj_type
        self.num_tokens = config.num_tokens
        self.output_hidden_size = config.output_hidden_size  # LM hidden size

        if self.proj_type.startswith("c-abs") and isinstance(self.proj_config, dict):
            self.proj_config = HoneybeeVisualProjectorConfig(**self.proj_config)

        self.abstractor = {
            "c-abs": CAbstractor,
            "d-abs": DAbstractor,
        }[
            self.proj_type
        ](self.proj_config, self.num_tokens, self.output_hidden_size)

        # deformable attention only supports fp32
        if type(self.abstractor) == DAbstractor:
            self.abstractor.to(torch.float)

        # Here, weights of abstractor (HoneybeeVisualProjectorModel) is initialized
        self.post_init()

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.abstractor.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HoneybeeProjectorModel):
            module.gradient_checkpointing = value

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            if self.proj_type == "d-abs":
                layer_outputs = torch.utils.checkpoint.checkpoint(self.abstractor, x)["last_hidden_state"]
            else:
                layer_outputs = torch.utils.checkpoint.checkpoint(self.abstractor, x)
        else:
            if self.proj_type == "d-abs":
                layer_outputs = self.abstractor(x)["last_hidden_state"]
            else:
                layer_outputs = self.abstractor(x)
        return layer_outputs

