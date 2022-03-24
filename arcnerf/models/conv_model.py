# -*- coding: utf-8 -*-

from common.models.base_model import BaseModel
from common.models.compents import ConvBNRelu
from common.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ConvModel(BaseModel):
    """Simple Conv Model"""

    def __init__(self, cfgs):
        super(ConvModel, self).__init__(cfgs)
        self.conv = ConvBNRelu(in_channels=cfgs.model.in_channel, out_channels=cfgs.model.out_channel)

    def forward(self, x):
        output = {'img': self.conv(x)}

        return output
