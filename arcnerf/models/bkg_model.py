# -*- coding: utf-8 -*-

import torch

from arcnerf.render.ray_helper import get_zvals_outside_sphere
from common.utils.cfgs_utils import get_value_from_cfgs_field
from .base_3d_model import Base3dModel


class BkgModel(Base3dModel):
    """Class for bkg model. Child class of Base3dModel.
    It can also be used as foreground model if you want.

    """

    def __init__(self, cfgs):
        super(BkgModel, self).__init__(cfgs)

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        raise NotImplementedError('Please implement the forward func...')

    def get_zvals_outside_sphere(self, rays_o: torch.Tensor, rays_d: torch.Tensor, inference_only=False):
        """Get the zvals from ray-sphere intersection.

        It will use ray_cfgs['n_sample'] to select samples.
                    ray_cfgs['bounding_radius'] as the inner sphere radius.
        Other sample keys are not allowed.

        Args:
            rays_o: torch.tensor (B, 1) near z distance
            rays_d: torch.tensor (B, 1) far z distance
            inference_only: If True, will not pertube the zvals. used in eval/infer model. Default False.

        Returns:
            zvals: torch.tensor (B, N_sample) zvlas of ray-sphere intersection
            radius: torch.tensor (B, N_sample) radius of each sphere
        """
        zvals, radius = get_zvals_outside_sphere(
            rays_o,
            rays_d,
            self.get_ray_cfgs('n_sample'),
            self.get_ray_cfgs('bounding_radius'),
            perturb=self.get_ray_cfgs('perturb') if not inference_only else False
        )  # (B, N_sample), (N_sample, )
        radius = torch.repeat_interleave(radius.unsqueeze(0).unsqueeze(-1), rays_o.shape[0], 0)  # (B, N_sample)

        return zvals, radius

    def read_optim_cfgs(self):
        """Read optim params under model.obj_bound. Prams controls optimization"""
        optim_cfgs = self.cfgs.model.optim
        params = {
            'near_distance': get_value_from_cfgs_field(optim_cfgs, 'near_distance', 0.0),
            'epoch_optim': get_value_from_cfgs_field(optim_cfgs, 'epoch_optim', 16),  # You must optimize the volume
            'epoch_optim_warmup': get_value_from_cfgs_field(optim_cfgs, 'epoch_optim_warmup', 256),
            'ema_optim_decay': get_value_from_cfgs_field(optim_cfgs, 'ema_optim_decay', 0.95),
            'opa_thres': get_value_from_cfgs_field(optim_cfgs, 'opa_thres', 0.01)
        }

        return params

    def get_optim_cfgs(self, key=None):
        """Get optim cfgs by optional key"""
        if key is None:
            return self.optim_cfgs

        return self.optim_cfgs[key]

    def set_optim_cfgs(self, key, value):
        """Set optim cfgs by key"""
        self.optim_cfgs[key] = value

    def optimize(self, cur_epoch=0):
        """Optimize the bkg structure. Support ['multivol'] now."""
        return
