# -*- coding: utf-8 -*-

import torch

from .base_modules import GeoNet, RadianceNet
from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from arcnerf.render.ray_helper import get_zvals_outside_sphere, ray_marching
from common.models.base_model import BaseModel
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing

__all__ = ['NeRFPP']


class BkgModel(BaseModel):
    """Class for bkg model. API Similar to base_3d_model so that build model can call.
     But do not inherit base_3d_model since base_3d_model calls its _forward() function to avoid circular import.
     """

    def __init__(self, cfgs):
        super(BkgModel, self).__init__(cfgs)
        # custom rays cfgs
        self.rays_cfgs = self.read_ray_cfgs()
        self.chunk_rays = self.cfgs.model.chunk_rays  # for n_rays together, do not consider n_pts on ray
        self.chunk_pts = self.cfgs.model.chunk_pts  # for n_pts together, only for model forward

    def read_ray_cfgs(self):
        """Read the ray cfgs and set default values"""
        ray_cfgs = {
            'bounding_radius': get_value_from_cfgs_field(self.cfgs.model.rays, 'bounding_radius'),
            'n_sample': get_value_from_cfgs_field(self.cfgs.model.rays, 'n_sample', 128),
            'inverse_linear': get_value_from_cfgs_field(self.cfgs.model.rays, 'inverse_linear', False),
            'perturb': get_value_from_cfgs_field(self.cfgs.model.rays, 'perturb', False),
            'add_inf_z': get_value_from_cfgs_field(self.cfgs.model.rays, 'add_inf_z', False),
            'noise_std': get_value_from_cfgs_field(self.cfgs.model.rays, 'noise_std', False),
            'white_bkg': get_value_from_cfgs_field(self.cfgs.model.rays, 'white_bkg', False),
        }

        return ray_cfgs

    def pretrain_siren(self):
        """Pretrain siren layer of implicit model"""
        self.geo_net.pretrain_siren()

    def forward(self, inputs, inference_only=False, get_progress=False):
        """The forward function actually call chunk process func ._forward()
        to avoid large memory at same time.
        Do not call this directly using chunk since the tensor are not flatten to represent batch of rays.

        Args:
            inputs: a dict of torch tensor:
                inputs['rays_o']: torch.tensor (B, N, 3), cam_loc/ray_start position
                inputs['rays_d']: torch.tensor (B, N, 3), view dir(assume normed)
                inputs['mask']: torch.tensor (B, N), mask value in {0, 1}. optional
                inputs['bounds']: torch.tensor (B, 2). optional
            inference_only: If True, only return the final results(not coarse). By default False
            get_progress: If True, output some progress for recording, can not used in inference only mode.
                          By default False

        Returns:
            output is a dict keys like (rgb, rgb_coarse, rgb_dense, depth, etc) based on the _forward function.
            If get_progress is True, output will contain keys like 'progress_xx' for xx in ['sigma', 'zvals'] etc.
        """
        flat_inputs = {}
        rays_o = inputs['rays_o'].view(-1, 3)  # (BN, 3)
        rays_d = inputs['rays_d'].view(-1, 3)  # (BN, 3)
        batch_size, n_rays_per_batch = inputs['rays_o'].shape[:2]

        flat_inputs['rays_o'] = rays_o
        flat_inputs['rays_d'] = rays_d

        # optional inputs
        bounds = None
        if 'bounds' in inputs:
            bounds = inputs['bounds'].view(-1, 2)  # (BN, 3)
        flat_inputs['bounds'] = bounds

        mask = None
        if 'mask' in inputs:
            mask = inputs['mask'].view(-1)  # (BN,)
        flat_inputs['mask'] = mask

        # all output tensor in (B*N, ...), reshape to (B, N, ...)
        output = chunk_processing(self._forward, self.chunk_rays, flat_inputs, inference_only, get_progress)
        for k, v in output.items():
            if isinstance(v, torch.Tensor) and batch_size * n_rays_per_batch == v.shape[0]:
                new_shape = tuple([batch_size, n_rays_per_batch] + list(v.shape)[1:])
                output[k] = v.view(new_shape)
            else:
                output[k] = v

        return output

    def _forward(self, inputs, inference_only=False, get_progress=False):
        """
        All the tensor are in chunk. B is total num of rays by grouping different samples in batch
        Args:
            inputs: a dict of torch tensor:
                inputs['rays_o']: torch.tensor (B, 3), cam_loc/ray_start position
                inputs['rays_d']: torch.tensor (B, 3), view dir(assume normed)
                inputs['mask']: torch.tensor (B,), mask value in {0, 1}. optional
            inference_only: If False, perturb the radius to increase robustness
            get_progress: If True, output some progress for recording, can not used in inference only mode.
                          By default False

        Returns:
            output is a dict with following keys:
                rgb: torch.tensor (B, 3)
                depth: torch.tensor (B,)
                mask: torch.tensor (B,)
        """
        raise NotImplementedError('Please implement the core forward function')

    @torch.no_grad()
    def forward_pts_dir(self, pts: torch.Tensor, view_dir: torch.Tensor = None):
        """This function forward pts and view dir directly, only for inference the geometry/color

        Args:
            pts: torch.tensor (N_pts, 3), pts in world coord
            view_dir: torch.tensor (N_pts, 3) view dir associate with each point. It can be normal or others.
                      If None, use (0, 0, 0) as the dir for each point.
        Returns:
            output is a dict with following keys:
                sigma/sdf: torch.tensor (N_pts), geometry value for each point
                rgb: torch.tensor (N_pts, 3), color for each point
        """
        sigma, feature = chunk_processing(self.geo_net, self.chunk_pts, pts)
        if view_dir is None:
            rays_d = torch.zeros_like(pts, dtype=pts.dtype).to(pts.device)
        else:
            rays_d = normalize(view_dir)  # norm view dir
        rgb = chunk_processing(self.radiance_net, self.chunk_pts, pts, rays_d, None, feature)

        return sigma[..., 0], rgb

    @torch.no_grad()
    def forward_pts(self, pts: torch.Tensor):
        """This function forward pts directly, only for inference the geometry

        Args:
            pts: torch.tensor (N_pts, 3), pts in world coord

        Returns:
            output is a dict with following keys:
                sigma/sdf: torch.tensor (N_pts), geometry value for each point
        """
        sigma, _ = chunk_processing(self.geo_net, self.chunk_pts, pts)

        return sigma[..., 0]


@MODEL_REGISTRY.register()
class NeRFPP(BkgModel):
    """ Nerf++ model. 8 layers in GeoNet and 1 layer in RadianceNet.
     Process bkg points only. Do not support geometric extractration.
        ref: https://arxiv.org/abs/2010.07492
    """

    def __init__(self, cfgs):
        super(NeRFPP, self).__init__(cfgs)
        self.geo_net = GeoNet(**self.cfgs.model.geometry.__dict__)
        self.radiance_net = RadianceNet(**self.cfgs.model.radiance.__dict__)

    def read_ray_cfgs(self):
        """Read the ray cfgs and set default values"""
        ray_cfgs = super().read_ray_cfgs()
        assert ray_cfgs['bounding_radius'] is not None, 'Please specify the bounding radius for nerf++ model'

        return ray_cfgs

    def _forward(self, inputs, inference_only=False, get_progress=False):
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)

        # get zvals for background, intersection from Multi-Sphere(MSI) (B, N_sample)
        zvals, radius = get_zvals_outside_sphere(
            rays_o,
            rays_d,
            self.rays_cfgs['n_sample'],
            self.rays_cfgs['bounding_radius'],
            perturb=self.rays_cfgs['perturb'] if not inference_only else False
        )  # (B, N_sample), (N_sample, )
        radius = torch.repeat_interleave(radius.unsqueeze(0).unsqueeze(-1), rays_o.shape[0], 0)  # (B, N_sample)

        # get points and change to (x/r, y/r, z/r, 1/r). Only when rays_o is (0,0,0) all points xyz norm as same.
        pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (B, N_sample, 3)
        pts = torch.cat([pts / radius, 1 / radius], dim=-1)
        pts = pts.view(-1, 4)  # (B*N_sample, 4)

        # get sigma and rgb,  expand rays_d to all pts. shape in (B*N_sample, dim)
        sigma, feature = chunk_processing(self.geo_net, self.chunk_pts, pts)
        rays_d_repeat = torch.repeat_interleave(rays_d, self.rays_cfgs['n_sample'], dim=0)
        radiance = chunk_processing(self.radiance_net, self.chunk_pts, pts, rays_d_repeat, None, feature)

        # reshape, ray marching and get color/weights
        sigma = sigma.view(-1, self.rays_cfgs['n_sample'], 1)[..., 0]  # (B, N_sample)
        radiance = radiance.view(-1, self.rays_cfgs['n_sample'], 3)  # (B, N_sample, 3)

        # ray marching. If two stage and inference only, get weights from single stage.
        output = ray_marching(
            sigma,
            radiance,
            zvals,
            self.rays_cfgs['add_inf_z'],
            self.rays_cfgs['noise_std'] if not inference_only else 0.0,
            weights_only=False,
            white_bkg=self.rays_cfgs['white_bkg']
        )

        if get_progress:  # this save the sigma/radiance with out blending bkg
            for key in ['sigma', 'zvals', 'alpha', 'trans_shift', 'weights', 'radiance']:
                output['progress_{}'.format(key)] = output[key].detach()  # (B, N_sample-1)

        output['rgb'] = output['rgb']  # (B, 3)
        output['depth'] = output['depth']  # (B,)
        output['mask'] = output['mask']  # (B,)

        return output
