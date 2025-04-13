# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Union

from transformers import PretrainedConfig
from transformers.models.deformable_detr import DeformableDetrConfig


class HoneybeeVisualProjectorConfig(PretrainedConfig):
    model_type = "mllm_visual_projector"

    def __init__(
        self,
        projector_type: str = "resampler",
        hidden_size: int = 1024,  #
        num_hidden_layers: int = 6,  #
        num_attention_heads: int = 16,  #
        intermediate_size: int = 4096,  #
        attention_probs_dropout_prob: float = 0.1,  #
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-6,  #
        encoder_hidden_size: int = 1024,  # This will be overwritten by vision_model's hidden_size
        pos_emb=False,
        feature_layer_index=-1,  # vision feature layer index; -1: last layer
        num_eos_tokens=1,
        use_cls=True,
        prenorm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder_hidden_size = encoder_hidden_size

        self.pos_emb = pos_emb
        self.feature_layer_index = feature_layer_index
        self.num_eos_tokens = num_eos_tokens
        self.use_cls = use_cls
        self.prenorm = prenorm

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the visual_projector config dict if we are loading from HoneybeeConfig
        if config_dict.get("model_type") == "mllm":
            config_dict = config_dict["projector_config"]

        # if (
        #     "model_type" in config_dict
        #     and hasattr(cls, "model_type")
        #     and config_dict["model_type"] != cls.model_type
        # ):
        #     logger.warning(
        #         f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
        #         f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
        #     )

        return cls.from_dict(config_dict, **kwargs)


class HoneybeeProjectorConfig(PretrainedConfig):
    model_type = 'honeybee_projector'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        config_path='/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/honeybee_configs/Honeybee-C-7B-M144-config.json',
        vit_num_tokens=577,
        num_queries=144,
        visual_hidden_size=4096,
        llm_hidden_size=4096,
        cascaded=False,
        num_sub_images=0,
        **kwargs,
    ):
        visual_projector_config = PretrainedConfig.from_pretrained(config_path).visual_projector_config

        if visual_projector_config["projector_type"].startswith("d-abs"):
            self.visual_projector_config = DeformableDetrConfig()
            self.visual_projector_config.update(visual_projector_config)
        else:
            self.visual_projector_config = HoneybeeVisualProjectorConfig(**visual_projector_config)

        if self.visual_projector_config.encoder_hidden_size != visual_hidden_size:
            self.visual_projector_config.encoder_hidden_size = visual_hidden_size
    

        if self.visual_projector_config.num_queries != num_queries:
            self.visual_projector_config.num_queries = num_queries


        # self.visual_projector_config.cascaded = cascaded
        # self.visual_projector_config.num_sub_images = num_sub_images

        # if self.visual_projector_config.projector_type == 'c-abs':
        #     self.visual_projector_config.pos_emb = False

        self.proj_type = self.visual_projector_config.projector_type
        self.num_tokens = vit_num_tokens
        self.output_hidden_size = llm_hidden_size  # LM hidden size
        super().__init__(**kwargs)
