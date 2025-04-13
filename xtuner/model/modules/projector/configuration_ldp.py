# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig


class LDPProjectorConfig(PretrainedConfig):
    model_type = 'ldp_projector'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        visual_hidden_size=4096,
        llm_hidden_size=4096,
        dw_size=(12, 12),
        **kwargs,
    ):
        self.visual_hidden_size = visual_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.dw_size = dw_size
        super().__init__(**kwargs)
