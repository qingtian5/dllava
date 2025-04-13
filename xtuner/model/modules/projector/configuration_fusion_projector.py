# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig

class FusionProjectorConfig(PretrainedConfig):
    model_type = 'fusion_projector'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        visual_hidden_size=4096,
        llm_hidden_size=4096,
        hidden_size=1024,
        num_reg_queries=64,
        num_ca_queries=64,
        vit_num_tokens=1024,
        num_feats=576,
        num_heads=1024 // 128,
        hws=None,
        reg_depth=3,
        mlp_depth=2,
        hb_visual_projector_config='/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/honeybee_configs/Honeybee-C-7B-M144-config.json',
        mlp_act_func='silu',
        projector_type='',
        channel_fusion=False,
        channel_fusion_hs1=None,
        channel_fusion_hs2=None,
        **kwargs,
    ):
        self.visual_hidden_size = visual_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.hidden_size = hidden_size
        self.vit_num_tokens = vit_num_tokens
        self.hws = hws
        self.num_reg_queries = num_reg_queries
        self.num_ca_queries = num_ca_queries
        self.num_feats = num_feats
        self.num_heads = num_heads
        self.reg_depth = reg_depth
        self.mlp_depth = mlp_depth
        self.mlp_act_func = mlp_act_func
        self.projector_type=projector_type
        self.channel_fusion=channel_fusion
        self.channel_fusion_hs1=channel_fusion_hs1
        self.channel_fusion_hs2=channel_fusion_hs2

        super().__init__(**kwargs)
