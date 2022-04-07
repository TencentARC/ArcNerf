# -*- coding: utf-8 -*-

from .base_modules import GeoNet, RadianceNet
from common.models.base_model import BaseModel
from common.utils.cfgs_utils import get_value_from_cfgs_field
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
        # below for rays
        self.rays_chunk = self.cfgs.model.rays_chunk
        self.bounding_radius = get_value_from_cfgs_field(self.cfgs.rays, 'bounding_radius')
        self.near = get_value_from_cfgs_field(self.cfgs.rays, 'near')
        self.far = get_value_from_cfgs_field(self.cfgs.rays, 'far')

    def forward(self, inputs):
        """
        Args:
            inputs['x']: torch.tensor (B, 3), xyz position
            inputs['view_dirs']: torch.tensor (B, 3), view dir(normed)

        Returns:
            sigma: torch.tensor (B, 1)
            rgb: torch.tensor (B, 3)
        """
        sigma, feature = self.geo_net(inputs['x'])
        rgb = self.radiance_net(None, inputs['view_dirs'], None, feature)

        return sigma, rgb
