from transformers import PretrainedConfig


class UHDProjectorConfig(PretrainedConfig):
    model_type = 'uhd_projector'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        num_queries=64,
        embed_dim=4096,
        num_heads=16,
        kv_dim=4096,
        tgt_size=(32,32),
        **kwargs,
    ):
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kv_dim = kv_dim
        self.tgt_size = tgt_size
        super().__init__(**kwargs)
