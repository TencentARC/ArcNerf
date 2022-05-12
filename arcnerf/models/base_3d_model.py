# -*- coding: utf-8 -*-

import torch

from arcnerf.geometry.transformation import normalize
from arcnerf.render.ray_helper import get_near_far_from_rays, get_zvals_from_near_far, ray_marching
from common.models.base_model import BaseModel
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.torch_utils import chunk_processing


class Base3dModel(BaseModel):
    """Base model for 3d reconstruction. Either used for foreground or background. Detail should be in child class """

    def __init__(self, cfgs):
        super(Base3dModel, self).__init__(cfgs)
        # ray_cfgs
        self.ray_cfgs = self.read_ray_cfgs()
        self.chunk_rays = self.cfgs.model.chunk_rays  # for n_rays together, do not consider n_pts on ray
        self.chunk_pts = self.cfgs.model.chunk_pts  # for n_pts together, only for model forward
        # set add_inf_z from cfgs
        self.add_inf_z = self.get_ray_cfgs('add_inf_z')

    def set_add_inf_z(self, add_inf_z):
        """hard set add_inf_z"""
        self.add_inf_z = add_inf_z

    @staticmethod
    def sigma_reverse():
        """Whether use sigma(inside object is large)
                       or sigma_reverse(inside objet is smaller, like sdf)
          By default False(Use sigma density)
        """
        return False

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
            'white_bkg': get_value_from_cfgs_field(self.cfgs.model.rays, 'white_bkg', False),
        }
        return ray_cfgs

    def get_ray_cfgs(self, key=None):
        if key is None:
            return self.ray_cfgs

        return self.ray_cfgs[key]

    def get_chunk_rays(self):
        """Get the chunk rays num"""
        return self.chunk_rays

    def get_chunk_pts(self):
        """Get the chunk pts num"""
        return self.chunk_pts

    def set_chunk_rays(self, chunk_rays):
        """Set the chunk rays num"""
        self.chunk_rays = chunk_rays

    def set_chunk_pts(self, chunk_pts):
        """Set the chunk pts num"""
        self.chunk_pts = chunk_pts

    def pretrain_siren(self):
        """Pretrain siren layer of implicit model.
        Need to rewrite if your network name is different
        """
        self.geo_net.pretrain_siren()

    def get_near_far_from_rays(self, inputs):
        """Get the near/far zvals from rays given settings

        Args:
            inputs: a dict of torch tensor:
                rays_o: torch.tensor (B, 3), cam_loc/ray_start position
                rays_d: torch.tensor (B, 3), view dir(assume normed)
                bounds: torch.tensor (B, 2). optional
            Returns:
                near, far:  torch.tensor (B, 1) each
        """
        bounds = None
        if 'bounds' in inputs:
            bounds = inputs['bounds'] if 'bounds' in inputs else None
        near, far = get_near_far_from_rays(
            inputs['rays_o'], inputs['rays_d'], bounds, self.get_ray_cfgs('near'), self.get_ray_cfgs('far'),
            self.get_ray_cfgs('bounding_radius')
        )

        return near, far

    def get_zvals_from_near_far(self, near: torch.Tensor, far: torch.Tensor, inference_only=False):
        """Get te zvals from near/far.
        It will use ray_cfgs['n_sample'] to select coarse samples.
        Other sample keys are not allowed.

        Args:
            near: torch.tensor (B, 1) near z distance
            far: torch.tensor (B, 1) far z distance
            inference_only: If True, will not pertube the zvals. used in eval/infer model. Default False.

        Returns:
            zvals: torch.tensor (B, N_sample)
        """
        zvals = get_zvals_from_near_far(
            near,
            far,
            self.get_ray_cfgs('n_sample'),
            inverse_linear=self.get_ray_cfgs('inverse_linear'),
            perturb=self.get_ray_cfgs('perturb') if not inference_only else False
        )  # (B, N_sample)

        return zvals

    def ray_marching(
        self,
        sigma: torch.Tensor,
        radiance: torch.Tensor,
        zvals: torch.Tensor,
        add_inf_z: bool = None,
        alpha: torch.Tensor = None,
        inference_only=False,
        weights_only=False
    ):
        """Ray marching and get output

        It will use self.add_inf_z to blend inf depth.
                    ray_cfgs['noise_std'] to add noise to sigma

        Other sample keys are not allowed.

        Args:
            sigma: (B, N_pts), density value, can use alpha directly. optional if alpha is input
            radiance: (B, N_pts, 3), radiance value for each point. If none, will not cal rgb from weighted radiance
            zvals: (B, N_pts), zvals for ray in unit-length
            add_inf_z: If not None, force the model to do ray_marching adding inf zvals.
                       Only will use it when call this function outside. By default None.
            alpha: (B, N_pts) In some model, it generates alpha directly instead of sigma(Neus).
                    Allow directly blending. By default None.
            inference_only: If True, will not pertube the zvals. used in eval/infer model. Default False.
            weights_only: Return weights only if True. By default False.

        Returns:
            output a dict with following keys:
                rgb/depth/mask in (B, ...), sigma/zvals/radiance/alpha/trans_shift/weights in (B, N_pts(-1), ...)
        """
        output = ray_marching(
            sigma,
            radiance,
            zvals,
            self.add_inf_z if add_inf_z is None else add_inf_z,  # can be set from outside
            self.get_ray_cfgs('noise_std') if not inference_only else 0.0,
            weights_only=weights_only,
            white_bkg=self.get_ray_cfgs('white_bkg'),
            alpha=alpha
        )

        return output

    def output_get_progress(self, output, get_progress=False, n_fg=None):
        """Keep the progress or pop it in the output from ray_marching
        Detach and rename the keys

        Args:
            output: is dict with:
                rgb/depth/mask in (B, ...), sigma/zvals/radiance/alpha/trans_shift/weights in (B, N_pts(-1), ...)
            get_progress:  if True, keep sigma/zvals/radiance/... for progress record.
                           else pop them to save memory
            n_fg: If not None, select n_fg pts only for all progress.
                  This happens when you blend sigma from bkg and run raymarching again.
        """
        if get_progress:  # rename the keys
            for key in ['sigma', 'zvals', 'alpha', 'trans_shift', 'weights', 'radiance']:
                if n_fg is not None:
                    output['progress_{}'.format(key)] = output[key][:, n_fg]  # (B, N_fg)
                else:
                    output['progress_{}'.format(key)] = output[key]  # (B, N_sample(-1))
            if self.sigma_reverse():
                output['progress_sigma_reverse'] = True  # for rays 3d visual of sdf

        # pop useless keys to reduce memory
        for key in ['sigma', 'zvals', 'alpha', 'trans_shift', 'weights', 'radiance']:
            output.pop(key)

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

    def forward_pts_dir(self, pts: torch.Tensor, view_dir: torch.Tensor = None):
        """This function forward pts and view dir directly, only for inference the geometry/color
        Assert you only have geo_net and radiance_net

        Args:
            pts: torch.tensor (N_pts, 3), pts in world coord
            view_dir: torch.tensor (N_pts, 3) view dir associate with each point. It can be normal or others.
                      If None, use (0, 0, 0) as the dir for each point.
        Returns:
            output: is a dict with following keys:
                sigma/sdf: torch.tensor (N_pts), geometry value for each point
                rgb: torch.tensor (N_pts, 3), color for each point
        """
        if view_dir is None:
            rays_d = torch.zeros_like(pts, dtype=pts.dtype).to(pts.device)
        else:
            rays_d = normalize(view_dir)  # norm view dir

        sigma, rgb = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, self.geo_net, self.radiance_net, pts, rays_d
        )

        return sigma, rgb

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
            rays_d: (B, 3) view dir(normalize)

        Return:
            sigma: (B, ) sigma value
            radiance: (B, 3) rgb value in float
        """
        sigma, feature = geo_net(pts)
        radiance = radiance_net(pts, rays_d, None, feature)

        return sigma[..., 0], radiance

    def forward_pts(self, pts: torch.Tensor):
        """This function forward pts directly, only for inference the geometry
        Assert you only have geo_net and radiance_net

        Args:
            pts: torch.tensor (N_pts, 3), pts in world coord

        Returns:
            output: is a dict with following keys:
                sigma/sdf: torch.tensor (N_pts), geometry value for each point
        """
        sigma, _ = chunk_processing(self.geo_net, self.chunk_pts, False, pts)

        return sigma[..., 0]

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        """
        All the tensor are in chunk. B is total num of rays by grouping different samples in batch
        The inputs are flatten into (B, ...) from FullModel's (B, N_rays, ...)
        Args:
            inputs: a dict of torch tensor:
                inputs['rays_o']: torch.tensor (B, 3), cam_loc/ray_start position
                inputs['rays_d']: torch.tensor (B, 3), view dir(assume normed)
                inputs['mask']: torch.tensor (B,), mask value in {0, 1}. optional
                inputs['bounds']: torch.tensor (B, 2). optional
            inference_only: If True, will not output coarse results. By default False
            get_progress: If True, output some progress for recording, can not used in inference only mode.
                          By default False
            cur_epoch: current epoch, for training purpose only. By default 0.
            total_epoch: total num of epoch, for training purpose only. By default 300k.

        Returns:
            output: is a dict with following keys:
                coarse_rgb: torch.tensor (B, 3), only if inference_only=False
                coarse_depth: torch.tensor (B,), only if inference_only=False
                coarse_mask: torch.tensor (B,), only if inference_only=False
                Return bellow if inference_only
                    fine_rgb: torch.tensor (B, 3)
                    fine_depth: torch.tensor (B,)
                    fine_mask: torch.tensor (B,)
                If get_progress is True:
                    sigma/zvals/alpha/trans_shift/weights: torch.tensor (B, n_pts)
                    Use from fine stage if n_importance > 0
        """
        raise NotImplementedError('Please implement the core forward function')
