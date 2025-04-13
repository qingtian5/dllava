# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig


class TokenPackerConfig(PretrainedConfig):
    model_type = 'tokenpacker_projector'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        visual_hidden_size=4096,
        llm_hidden_size=4096,
        raw_grid=24,
        scale_factor=2,
        embed_dim=1024,
        num_heads=1024//128,
        **kwargs,
    ):
        self.visual_hidden_size = visual_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.raw_grid = raw_grid
        self.scale_factor = scale_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        super().__init__(**kwargs)
