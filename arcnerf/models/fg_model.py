# -*- coding: utf-8 -*-

import torch

from .base_3d_model import Base3dModel
from arcnerf.geometry.sphere import Sphere
from arcnerf.geometry.volume import Volume
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class FgModel(Base3dModel):
    """Class for fg model. Child class of Base3dModel
     You can use contain a explicit structure for fast/accurate inner object sampling and ray marching.
     The structure can be a dense volume, octree, sphere, or other structure etc.

     But by default it do not use such bounding structure, and the sampling is in a larger area.

     Any other modeling methods(NeRF, NeuS, mip-nerf) inherit this model and have their detailed sampling/rendering
     algorithms.
    """

    def __init__(self, cfgs):
        super(FgModel, self).__init__(cfgs)
        # inner object bounding structure
        self.obj_bound = get_value_from_cfgs_field(self.cfgs.model, 'obj_bound', None)
        self.obj_bound_type = None
        if self.obj_bound is not None:
            self.set_up_obj_bound_structure()
        # bkg color/depth/normal for invalid rays
        self.bkg_color = get_value_from_cfgs_field(self.cfgs.model, 'bkg_color', [1.0, 1.0, 1.0])  # white
        self.depth_far = get_value_from_cfgs_field(self.cfgs.model, 'depth_far', 10.0)  # far distance
        self.normal = get_value_from_cfgs_field(self.cfgs.model, 'normal', [1.0, 0.0, 0.0])  # for eikonal loss cal
        # optimize the struct(pruning the volume, etc) periodically
        self.epoch_optim = get_value_from_cfgs_field(self.cfgs.model, 'epoch_optim', None)

    def set_up_obj_bound_structure(self):
        """Set up the bounding structure of the model"""
        if 'volume' in self.obj_bound.__dict__.keys():
            volume_cfgs = get_value_from_cfgs_field(self.obj_bound, 'volume', None)
            if volume_cfgs is not None:
                self.obj_bound_type = 'volume'
                self.obj_bound = Volume(**volume_cfgs.__dict__)
        elif 'sphere' in self.obj_bound.__dict__.keys():
            sphere_cfgs = get_value_from_cfgs_field(self.obj_bound, 'sphere', None)
            if sphere_cfgs is not None:
                self.obj_bound_type = 'sphere'
                self.obj_bound = Sphere(**sphere_cfgs.__dict__)
        else:
            raise NotImplementedError('Obj bound type {} not support...'.format(self.obj_bound.__dict__.keys()))

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        """If you use a geometric structure bounding the object, some rays does not hit the bound can be ignored.
         You can assign a bkg color to them directly, with opacity=0.0 and depth=some far distance.
        """
        # optimize the obj bound, mostly for volume base structure
        if cur_epoch > 0 and self.epoch_optim is not None and cur_epoch % self.epoch_optim == 0:
            self.optim_obj_bound(cur_epoch)

    def optim_obj_bound(self, cur_epoch):
        """Optimize the obj bounding geometric structure. Support ['volume'] now."""
        if self.obj_bound_type == 'volume':
            pass

    def get_near_far_from_rays(self, inputs):
        """Get the near/far zvals from rays given settings. If the inner structure for model exists,
        it will sample only in the object area. Otherwise call Base3dModel's method to sample in large area.

        Args:
            inputs: a dict of torch tensor:
                rays_o: torch.tensor (B, 3), cam_loc/ray_start position
                rays_d: torch.tensor (B, 3), view dir(assume normed)
                bounds: torch.tensor (B, 2). optional
        Returns:
            near, far:  torch.tensor (B, 1) each
        """
        if self.obj_bound_type is not None:
            if self.obj_bound_type == 'volume':
                near, far, _, _ = self.obj_bound.ray_volume_intersection(inputs['rays_o'], inputs['rays_d'])
            elif self.obj_bound_type == 'sphere':
                near, far, _, _ = self.obj_bound.ray_sphere_intersection(inputs['rays_o'], inputs['rays_d'])
            else:
                raise NotImplementedError('Ray-{} is not valid...Please implement it...'.format(self.obj_bound_type))
        else:  # call the parent class
            near, far = super().get_near_far_from_rays(inputs)

        return near, far

    def get_zvals_from_near_far(self, near: torch.Tensor, far: torch.Tensor, n_pts, inference_only=False):
        """Get the zvals of the object with/without bounding structure"""
        zvals = None
        if self.obj_bound_type is not None:
            if self.obj_bound_type == 'volume':
                print('Volume based resampling')
                pass

        if zvals is None:
            zvals = super().get_zvals_from_near_far(near, far, n_pts, inference_only)

        return zvals
