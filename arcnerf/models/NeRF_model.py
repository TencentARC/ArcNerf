# -*- coding: utf-8 -*-

from .base_modules import GeoNet, RadianceNet
from common.models.base_model import BaseModel
from common.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class NeRF(BaseModel):
    """Single forward Nerf model. 8 layers in GeoNet and 1 layer in RadianceNet
    ref: https://www.matthewtancik.com/nerf
    """

    def __init__(self, cfgs):
        super(NeRF, self).__init__(cfgs)
        self.geo_net = GeoNet(**self.cfgs.model.geometry.__dict__)
        self.radiance_net = RadianceNet(**self.cfgs.model.radiance.__dict__)
        self.rays_chunk = self.cfgs.model.rays_chunk

    def forward(self, inputs):
        """
        Args:
            inputs['x']: torch.tensor (B, 3)
            inputs['view_dirs']: torch.tensor (B, 3)

        Returns:
            sigma: torch.tensor (B, 1)
            rgb: torch.tensor (B, 3)
        """
        sigma, feature = self.geo_net(inputs['x'])
        rgb = self.radiance_net(None, inputs['view_dirs'], None, feature)

        return sigma, rgb
