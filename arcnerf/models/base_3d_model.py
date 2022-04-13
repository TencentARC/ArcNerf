# -*- coding: utf-8 -*-

from common.models.base_model import BaseModel
from common.utils.cfgs_utils import get_value_from_cfgs_field


class Base3dModel(BaseModel):
    """Base model for 3d reconstruction, mainly for reading cfgs. """

    def __init__(self, cfgs):
        super(Base3dModel, self).__init__(cfgs)

    def read_ray_cfgs(self):
        """Read cfgs for ray, common case"""
        ray_cfgs = {
            'bounding_radius': get_value_from_cfgs_field(self.cfgs.model.rays, 'bounding_radius'),
            'near': get_value_from_cfgs_field(self.cfgs.model.rays, 'near'),
            'far': get_value_from_cfgs_field(self.cfgs.model.rays, 'far'),
            'n_sample': get_value_from_cfgs_field(self.cfgs.model.rays, 'n_sample', 128),
            'inverse_linear': get_value_from_cfgs_field(self.cfgs.model.rays, 'inverse_linear', False),
            'perturb': get_value_from_cfgs_field(self.cfgs.model.rays, 'perturb', False),
            'add_inf_z': get_value_from_cfgs_field(self.cfgs.model.rays, 'add_inf_z', False),
            'noise_std': get_value_from_cfgs_field(self.cfgs.model.rays, 'noise_std', False),
        }
        return ray_cfgs

    def forward(self, x):
        raise NotImplementedError('Please implement the forward func...')
