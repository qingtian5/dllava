# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.models.idefics2.modeling_idefics2 import Idefics2MLP, Idefics2PerceiverResampler

from .configuration_idefics2 import Idefics2ProjectorConfig


class Idefics2Connector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.modality_projection = Idefics2MLP(
            hidden_size=config.vision_config.hidden_size,
            intermediate_size=config.text_config.intermediate_size,
            output_size=config.text_config.hidden_size,
            hidden_act=config.text_config.hidden_act,
        )
        self.perceiver_resampler = Idefics2PerceiverResampler(config)

    def forward(self, image_hidden_states):
        attention_mask = torch.ones(([image_hidden_states.shape[0], image_hidden_states.shape[1]]), dtype=torch.bool, device=image_hidden_states.device)
        # print(f"1 image_hidden_states: {image_hidden_states.shape}")
        image_hidden_states = self.modality_projection(image_hidden_states)
        # print(f"2 image_hidden_states: {image_hidden_states.shape}")
        image_hidden_states = self.perceiver_resampler(context=image_hidden_states, attention_mask=attention_mask)
        # print(f"3 image_hidden_states: {image_hidden_states.shape}")
        return image_hidden_states


class Idefics2ProjectorModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = Idefics2ProjectorConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: Idefics2ProjectorConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False
        self.model = Idefics2Connector(config)

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.model.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Idefics2ProjectorConfig):
            module.gradient_checkpointing = value

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(self.model, x)
        else:
            layer_outputs = self.model(x)
        return layer_outputs
