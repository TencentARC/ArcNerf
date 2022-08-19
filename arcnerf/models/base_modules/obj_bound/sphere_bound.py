# -*- coding: utf-8 -*-

from . import BOUND_REGISTRY
from .basic_bound import BasicBound
from arcnerf.geometry.sphere import Sphere
from common.utils.cfgs_utils import valid_key_in_cfgs


@BOUND_REGISTRY.register()
class SphereBound(BasicBound):
    """A sphere structure bounding the object"""

    def __init__(self, cfgs):
        super(SphereBound, self).__init__(cfgs)
        assert valid_key_in_cfgs(cfgs, 'sphere'), 'You must have sphere in the cfgs'

        self.cfgs = cfgs
        self.read_optim_cfgs()

        # set up the sphere
        sphere_cfgs = cfgs.sphere
        self.sphere = Sphere(**sphere_cfgs.__dict__)

    def get_obj_bound(self):
        """Get the real obj bounding structure"""
        return self.sphere

    def get_near_far_from_rays(self, inputs, **kwargs):
        """Get the near/far zvals from rays using sphere bounding.

        Returns:
            near, far: torch.tensor (B, 1) each
            mask_rays: torch.tensor (B,), each rays validity
        """
        near, far, _, mask_rays = self.sphere.ray_sphere_intersection(inputs['rays_o'], inputs['rays_d'])

        return near, far, mask_rays[:, 0]
