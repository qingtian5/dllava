# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig

class DualPathProjectorConfig(PretrainedConfig):
    model_type = 'dualpath_projector'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        config_thumbnail=None, 
        config_tiles=None,
        **kwargs,
    ):
        self.config_thumbnail = config_thumbnail
        self.config_tiles = config_tiles
        super().__init__(**kwargs)
