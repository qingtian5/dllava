# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig, AutoModel

from .configuration_projector import ProjectorConfig
from .modeling_projector import ProjectorModel

from .configuration_dualpath_projector import DualPathProjectorConfig
from .modeling_dualpath_projector import DualPathProjectorModel

from .configuration_fusion_projector import FusionProjectorConfig
from .modeling_fusion_projector import FusionProjectorModel

from .configuration_honeybee import HoneybeeProjectorConfig
from .modeling_honeybee_projector import HoneybeeProjectorModel

from .configuration_ldp import LDPProjectorConfig
from .modeling_ldp_projector import LDPProjectorModel

from .configuration_uhd import UHDProjectorConfig
from .modeling_uhd_projector import UHDProjectorModel

from .configuration_idefics2 import Idefics2ProjectorConfig
from .modeling_idefics2_projector import Idefics2ProjectorModel

from .configuration_tokenpacker import TokenPackerConfig
from .modeling_tokenpacer_projector import TokenPackerProjectorModel

AutoConfig.register('fusion_projector', FusionProjectorConfig)
AutoModel.register(FusionProjectorConfig, FusionProjectorModel)

AutoConfig.register('projector', ProjectorConfig)
AutoModel.register(ProjectorConfig, ProjectorModel)

AutoConfig.register('dualpath_projector', DualPathProjectorConfig)
AutoModel.register(DualPathProjectorConfig, DualPathProjectorModel)

AutoConfig.register('ldp_projector', LDPProjectorConfig)
AutoModel.register(LDPProjectorConfig, LDPProjectorModel)

AutoConfig.register('uhd_projector', UHDProjectorConfig)
AutoModel.register(UHDProjectorConfig, UHDProjectorModel)

AutoConfig.register('idefics2_projector', Idefics2ProjectorConfig)
AutoModel.register(Idefics2ProjectorConfig, Idefics2ProjectorModel)

AutoConfig.register('honeybee_projector', HoneybeeProjectorConfig)
AutoModel.register(HoneybeeProjectorConfig, HoneybeeProjectorModel)

AutoConfig.register('tokenpacker_projector', TokenPackerConfig)
AutoModel.register(TokenPackerConfig, TokenPackerProjectorModel)

__all__ = ['ProjectorConfig', 'ProjectorModel', 
           'DualPathProjectorConfig', 'DualPathProjectorModel',
           'FusionProjectorConfig', 'FusionProjectorModel',
           'HoneybeeProjectorConfig', 'HoneybeeProjectorModel', 
           'LDPProjectorConfig', 'LDPProjectorModel', 
           'UHDProjectorConfig', 'UHDProjectorModel',
           'Idefics2ProjectorConfig', 'Idefics2ProjectorModel', 
           'TokenPackerConfig', 'TokenPackerProjectorModel']
