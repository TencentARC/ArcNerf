# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from arcnerf.render.ray_helper import get_near_far_from_rays, get_zvals_from_near_far
from common.utils.cfgs_utils import get_value_from_cfgs_field
from . import BOUND_REGISTRY


@BOUND_REGISTRY.register()
class BasicBound(nn.Module):
    """A bounding structure of the object"""

    def __init__(self, cfgs):
        super(BasicBound, self).__init__()
        self.cfgs = cfgs
        self.optim_cfgs = self.read_optim_cfgs()

    def get_obj_bound(self):
        """Get the real obj bounding structure"""
        return None

    def get_optim_cfgs(self, key=None):
        """Get optim cfgs by optional key"""
        if key is None:
            return self.optim_cfgs

        return self.optim_cfgs[key]

    def set_optim_cfgs(self, key, value):
        """Set optim cfgs by key"""
        self.optim_cfgs[key] = value

    def read_optim_cfgs(self):
        """Read optim params under model.obj_bound. Prams controls optimization"""
        params = {
            'epoch_optim': get_value_from_cfgs_field(self.cfgs, 'epoch_optim', None),
            'epoch_optim_warmup': get_value_from_cfgs_field(self.cfgs, 'epoch_optim_warmup', None),
            'ema_optim_decay': get_value_from_cfgs_field(self.cfgs, 'ema_optim_decay', 0.95),
            'opa_thres': get_value_from_cfgs_field(self.cfgs, 'opa_thres', 0.01)
        }

        return params

    def get_near_far_from_rays(self, inputs, near_hardcode=None, far_hardcode=None, bounding_radius=None):
        """Get the near/far zvals from rays given settings

        Args:
            inputs: a dict of torch tensor:
                rays_o: torch.tensor (B, 3), cam_loc/ray_start position
                rays_d: torch.tensor (B, 3), view dir(assume normed)
                bounds: torch.tensor (B, 2). optional
            near_hardcode: hardcode near
            far_hardcode: hardcode far
            bounding_radius: large bounding radius

        Returns:
            near, far:  torch.tensor (B, 1) each
            mask_rays: Always return None, each rays validity
        """
        bounds = None
        if 'bounds' in inputs:
            bounds = inputs['bounds'] if 'bounds' in inputs else None
        near, far = get_near_far_from_rays(
            inputs['rays_o'], inputs['rays_d'], bounds, near_hardcode, far_hardcode, bounding_radius
        )

        return near, far, None

    def get_zvals_from_near_far(
        self, near, far, n_pts, inference_only=False, inverse_linear=False, perturb=False, **kwargs
    ):
        """Get the zvals of the object with/without bounding structure

        It will use ray_cfgs['n_sample'] to select coarse samples.
        Other sample keys are not allowed.

        Args:
            near: torch.tensor (B, 1) near z distance
            far: torch.tensor (B, 1) far z distance
            n_pts: num of points for zvals sampling. It is generally the `rays.n_sample` in the model_cfgs.
            inference_only: If True, will not pertube the zvals. used in eval/infer model. Default False.
            inverse_linear: get inverse linear samples. By default False
            perturb: perturb the samples. Only in train

        Returns:
            zvals: torch.tensor (B, N_sample)
            mask_pts: Always return None, indicating the validity of each pts.
        """
        zvals = get_zvals_from_near_far(
            near, far, n_pts, inverse_linear=inverse_linear, perturb=perturb if not inference_only else False
        )  # (B, N_sample)

        return zvals, None

    @torch.no_grad()
    def optimize(self, cur_epoch=0, n_pts=128, get_est_opacity=None):
        """Optimize the inner structure. Only volume needs it

        Args:
            cur_epoch: current epoch. only process periodically
            n_pts: use to find the dt
            get_est_opacity: A function that process dt/pts to get the opacity at this pts.
        """
        return
