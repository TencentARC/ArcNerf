# -*- coding: utf-8 -*-

import torch

from common.models.base_model import BaseModel
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing
from simplenerf.geometry.ray import get_ray_points_by_zvals
from simplenerf.render.ray_helper import sample_pdf, get_zvals_from_near_far, ray_marching, get_near_far_from_rays
from .base_modules.base_network import GeoNet, RadianceNet


@MODEL_REGISTRY.register()
class NeRF(BaseModel):
    """ Nerf model.
        The two-stage nerf use coarse/fine models for different stage, instead of using just one.
        ref: https://www.matthewtancik.com/nerf
    """

    def __init__(self, cfgs):
        super(NeRF, self).__init__(cfgs)
        # ray_cfgs
        self.ray_cfgs = self.read_ray_cfgs()
        self.chunk_rays = self.cfgs.model.chunk_rays  # for n_rays together, do not consider n_pts on ray
        self.chunk_pts = self.cfgs.model.chunk_pts  # for n_pts together, only for model forward
        # models
        self.coarse_geo_net = GeoNet(**self.cfgs.model.geometry.__dict__)
        self.coarse_radiance_net = RadianceNet(**self.cfgs.model.radiance.__dict__)
        # custom rays cfgs
        self.ray_cfgs['n_importance'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_importance', 0)
        # set fine model if n_importance > 0
        if self.get_ray_cfgs('n_importance') > 0:
            self.fine_geo_net = GeoNet(**self.cfgs.model.geometry.__dict__)
            self.fine_radiance_net = RadianceNet(**self.cfgs.model.radiance.__dict__)

    def read_ray_cfgs(self):
        """Read cfgs for ray, common case"""
        ray_cfgs = {
            'near': get_value_from_cfgs_field(self.cfgs.model.rays, 'near'),
            'far': get_value_from_cfgs_field(self.cfgs.model.rays, 'far'),
            'n_sample': get_value_from_cfgs_field(self.cfgs.model.rays, 'n_sample', 128),
            'inverse_linear': get_value_from_cfgs_field(self.cfgs.model.rays, 'inverse_linear', False),
            'perturb': get_value_from_cfgs_field(self.cfgs.model.rays, 'perturb', False),
            'noise_std': get_value_from_cfgs_field(self.cfgs.model.rays, 'noise_std', False),
            'white_bkg': get_value_from_cfgs_field(self.cfgs.model.rays, 'white_bkg', False),
            'non_ndc_view_dirs': get_value_from_cfgs_field(self.cfgs.model.rays, 'non_ndc_view_dirs', False)
        }
        return ray_cfgs

    def get_ray_cfgs(self, key=None):
        """Get ray cfgs by optional key"""
        if key is None:
            return self.ray_cfgs

        return self.ray_cfgs[key]

    @staticmethod
    def prepare_flatten_inputs(inputs):
        """Prepare the inputs by flatten them from (B, N, ...) to (BN, ...)

        Args:
            inputs: a dict of torch tensor:
                inputs['rays_o']: torch.tensor (B, N, 3), cam_loc/ray_start position
                inputs['rays_d']: torch.tensor (B, N, 3), rays direction
                inputs['view_dirs']: torch.tensor (B, N, 3), view dirs in non-ndc space.
                inputs['mask']: torch.tensor (B, N), mask value in {0, 1}. optional
                inputs['bounds']: torch.tensor (B, N, 2). optional

        Returns:
            flatten_inputs:
                value in inputs flatten into (BN, ...)
        """
        flat_inputs = {}
        rays_o = inputs['rays_o'].view(-1, 3)  # (BN, 3)
        rays_d = inputs['rays_d'].view(-1, 3)  # (BN, 3)
        view_dirs = inputs['view_dirs'].view(-1, 3)  # (BN, 3)
        batch_size, n_rays_per_batch = inputs['rays_o'].shape[:2]

        flat_inputs['rays_o'] = rays_o
        flat_inputs['rays_d'] = rays_d
        flat_inputs['view_dirs'] = view_dirs

        # optional inputs
        bounds = None
        if 'bounds' in inputs:
            bounds = inputs['bounds'].view(-1, 2)  # (BN, 2)
        flat_inputs['bounds'] = bounds

        mask = None
        if 'mask' in inputs:
            mask = inputs['mask'].view(-1)  # (BN,)
        flat_inputs['mask'] = mask

        return flat_inputs, batch_size, n_rays_per_batch

    @staticmethod
    def reshape_output(output, batch_size, n_rays_per_batch):
        """Reshape flatten output from (BN, ...) into (B, N, ...) dim"""
        for k, v in output.items():
            if isinstance(v, torch.Tensor) and batch_size * n_rays_per_batch == v.shape[0]:
                new_shape = tuple([batch_size, n_rays_per_batch] + list(v.shape)[1:])
                output[k] = v.view(new_shape)
            else:
                output[k] = v

        return output

    def adjust_coarse_fine_output(self, output, inference_only=False):
        """Adjust the output if use two stage model with coarse/fine output

        Args:
            output: contains coarse/fine as keys with two stage outputs
            inference_only: If True, return one set of output with keys ending without '_coarse/_fine'.
                            Else, return both sets of output with keys ending in '_coarse/_fine'.
        """
        assert 'n_importance' in self.get_ray_cfgs(), 'Not valid for two stage model...'
        if inference_only:
            return output['fine'] if self.get_ray_cfgs('n_importance') > 0 else output['coarse']

        output_cf = {}
        for k, v in output['coarse'].items():
            output_cf['{}_coarse'.format(k)] = v
        if self.get_ray_cfgs('n_importance') > 0:
            for k, v in output['fine'].items():
                output_cf['{}_fine'.format(k)] = v

        return output_cf

    def forward(self, inputs, inference_only=False):
        # prepare flatten inputs
        flat_inputs, batch_size, n_rays_per_batch = self.prepare_flatten_inputs(inputs)

        # get output
        output = chunk_processing(self._forward, self.chunk_rays, False, flat_inputs, inference_only)

        # reshape values from (B*N, ...) to (B, N, ...)
        output = self.reshape_output(output, batch_size, n_rays_per_batch)

        return output

    def _forward(self, inputs, inference_only=False):
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        if self.get_ray_cfgs('non_ndc_view_dirs'):
            view_dirs = inputs['view_dirs']  # (B, 3)
        else:
            view_dirs = inputs['rays_d']  # (B, 3)
        output = {}

        # get bounds for object
        bounds = None
        if 'bounds' in inputs:
            bounds = inputs['bounds'] if 'bounds' in inputs else None
        near, far = get_near_far_from_rays(
            rays_o, rays_d, bounds, self.get_ray_cfgs('near'), self.get_ray_cfgs('far')
        )  # (B, 1) * 2

        # coarse model
        # get zvals
        zvals = get_zvals_from_near_far(
            near,
            far,
            self.get_ray_cfgs('n_sample'),
            inverse_linear=self.get_ray_cfgs('inverse_linear'),
            perturb=self.get_ray_cfgs('perturb') if not inference_only else False
        )  # (B, N_sample)

        # get points
        pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (B, N_sample, 3)
        pts = pts.view(-1, 3)  # (B*N_sample, 3)

        # get sigma and rgb, expand rays_d to all pts. shape in (B*N_sample, ...)
        view_dirs_repeat = torch.repeat_interleave(view_dirs, self.get_ray_cfgs('n_sample'), dim=0)
        sigma, radiance = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, self.coarse_geo_net, self.coarse_radiance_net, pts,
            view_dirs_repeat
        )

        # reshape
        sigma = sigma.view(-1, self.get_ray_cfgs('n_sample'))  # (B, N_sample)
        radiance = radiance.view(-1, self.get_ray_cfgs('n_sample'), 3)  # (B, N_sample, 3)

        # ray marching for coarse network, keep the coarse weights for next stage
        output['coarse'] = ray_marching(
            sigma,
            radiance,
            zvals,
            rays_d,
            self.get_ray_cfgs('noise_std') if not inference_only else 0.0,
            weights_only=False,
            white_bkg=self.get_ray_cfgs('white_bkg'),
        )
        coarse_weights = output['coarse']['weights']

        # fine model
        if self.get_ray_cfgs('n_importance') > 0:
            # get upsampled zvals
            zvals = self.upsample_zvals(zvals, coarse_weights, inference_only)
            n_total = self.get_ray_cfgs('n_sample') + self.get_ray_cfgs('n_importance')

            # get upsampled pts
            pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (B, N_total, 3)
            pts = pts.view(-1, 3)  # (B*N_total, 3)

            # get sigma and rgb, expand rays_d to all pts. shape in (B*N_total, ...)
            view_dirs_repeat = torch.repeat_interleave(view_dirs, n_total, dim=0)
            sigma, radiance = chunk_processing(
                self._forward_pts_dir, self.chunk_pts, False, self.fine_geo_net, self.fine_radiance_net, pts,
                view_dirs_repeat
            )

            # reshape
            sigma = sigma.view(-1, n_total)  # (B, n_total)
            radiance = radiance.view(-1, n_total, 3)  # (B, n_total, 3)

            # ray marching for fine network
            output['fine'] = ray_marching(
                sigma,
                radiance,
                zvals,
                rays_d,
                self.get_ray_cfgs('noise_std') if not inference_only else 0.0,
                weights_only=False,
                white_bkg=self.get_ray_cfgs('white_bkg'),
            )

        # adjust two stage output
        output = self.adjust_coarse_fine_output(output, inference_only)

        return output

    def upsample_zvals(self, zvals: torch.Tensor, weights: torch.Tensor, inference_only=True):
        """Upsample zvals if N_importance > 0

        Args:
            zvals: tensor (B, N_sample), coarse zvals for all rays
            weights: tensor (B, N_sample) (B, N_sample(-1))
            inference_only: affect the sample_pdf deterministic. By default False(For train)

        Returns:
            zvals: tensor (B, N_sample + N_importance), up-sample zvals near the surface
        """
        weights_coarse = weights[:, 1:self.get_ray_cfgs('n_sample')-1]  # (B, N_sample-2)
        zvals_mid = 0.5 * (zvals[..., 1:] + zvals[..., :-1])  # (B, N_sample-1)
        _zvals = sample_pdf(
            zvals_mid, weights_coarse, self.get_ray_cfgs('n_importance'),
            not self.get_ray_cfgs('perturb') if not inference_only else True
        ).detach()
        zvals, _ = torch.sort(torch.cat([zvals, _zvals], -1), -1)  # (B, N_sample+N_importance=N_total)

        return zvals

    @staticmethod
    def _forward_pts_dir(
        geo_net,
        radiance_net,
        pts: torch.Tensor,
        rays_d: torch.Tensor = None,
    ):
        """Core forward function to forward. Rewrite it if you have more inputs from geo_net to radiance_net
        Use chunk progress to call it will save memory for feature since it does not save intermediate result.

        Args:
            pts: (B, 3) xyz points
            rays_d: (B, 3) rays direction

        Return:
            sigma: (B, ) sigma value
            radiance: (B, 3) rgb value in float
        """
        sigma, feature = geo_net(pts)
        radiance = radiance_net(rays_d, feature)

        return sigma[..., 0], radiance
