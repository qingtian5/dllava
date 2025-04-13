from transformers import PretrainedConfig
from transformers.models.idefics2.configuration_idefics2 import Idefics2PerceiverConfig


class Idefics2ProjectorConfig(PretrainedConfig):
    model_type = 'idefics2_projector'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        vision_config,
        text_config,
        **kwargs,
    ):
        # # for evaluate
        # if isinstance(vision_config, dict):
        #     print('load vision_config from dict')
        #     vision_config = AutoConfig.from_pretrained(vision_config['_name_or_path'])
        # if isinstance(text_config, dict):
        #     print('load text_config from dict')
        #     text_config = AutoConfig.from_pretrained(text_config['_name_or_path'])

        self.vision_config = vision_config
        self.text_config = text_config
        self.perceiver_config = Idefics2PerceiverConfig()

        # print(f"self.perceiver_config: {self.perceiver_config}")

        super().__init__(**kwargs)
