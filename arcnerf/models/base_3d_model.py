# -*- coding: utf-8 -*-

import torch

from .bkg_model import NeRFPP
from arcnerf.geometry.transformation import normalize
from arcnerf.render.ray_helper import get_near_far_from_rays
from common.models.base_model import BaseModel
from common.utils.cfgs_utils import valid_key_in_cfgs, get_value_from_cfgs_field, dict_to_obj, obj_to_dict
from common.utils.torch_utils import chunk_processing


class Base3dModel(BaseModel):
    """Base model for 3d reconstruction, mainly for reading cfgs. """

    def __init__(self, cfgs):
        super(Base3dModel, self).__init__(cfgs)
        # ray_cfgs
        self.rays_cfgs = self.read_ray_cfgs()
        self.chunk_rays = self.cfgs.model.chunk_rays  # for n_rays together, do not consider n_pts on ray
        self.chunk_pts = self.cfgs.model.chunk_pts  # for n_pts together, only for model forward
        # background model
        self.bkg_cfgs = self.read_bkg_cfgs()
        self.bkg_model = None
        if self.bkg_cfgs is not None:
            # bkg_blend type, 'rgb' or 'sigma
            self.bkg_blend = get_value_from_cfgs_field(self.bkg_cfgs, 'bkg_blend', 'rgb')
            self.bkg_add_inf_z = get_value_from_cfgs_field(self.bkg_cfgs.rays, 'add_inf_z', 'False')
            self.bkg_model = self.setup_bkg_model()
            self.check_bkg_cfgs()

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

    def check_bkg_cfgs(self):
        """If bkg model is used, check for invalid cfgs"""
        # foreground model should not add add_inf_z if bkg_blend == 'rgb',
        if self.bkg_blend == 'rgb':
            assert self.rays_cfgs['add_inf_z'] is False, 'Do not add_inf_z for foreground'
            assert self.bkg_add_inf_z is True, 'Must use add_inf_z for background in rgb blending mode'
        elif self.bkg_blend == 'sigma':
            assert self.bkg_add_inf_z is False, 'Do not add_inf_z for background in sigma blending mode'
        else:
            raise NotImplementedError('Invalid bkg_blend type {}'.format(self.bkg_blend))

        # far distance should not exceed 2*bkg_bounding_radius
        max_far = 2.0 * self.bkg_cfgs.rays.bounding_radius
        assert self.rays_cfgs['far'] is None or self.rays_cfgs['far'] <= max_far,\
            'Do not set far exceed {}'.format(max_far)

    def is_cuda(self):
        """Check whether the model is on cuda"""
        return next(self.parameters()).is_cuda

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

    def read_bkg_cfgs(self):
        """Read cfgs for background. Return None if did not use."""
        if valid_key_in_cfgs(self.cfgs.model, 'background'):
            return self.cfgs.model.background

        return None

    def setup_bkg_model(self):
        """Set up a background model"""
        assert valid_key_in_cfgs(self.bkg_cfgs, 'type'), 'You did not specify the bkg model type...'
        bkg_model_cfgs = dict_to_obj({'model': obj_to_dict(self.bkg_cfgs)})
        if self.bkg_cfgs.type == 'NeRFPP':
            bkg_model = NeRFPP(bkg_model_cfgs)
        else:
            raise NotImplementedError('Method {} for bkg not support yet...'.format(self.bkg_cfgs.type))

        return bkg_model

    def pretrain_siren(self):
        """Pretrain siren layer of implicit model"""
        self.geo_net.pretrain_siren()
        if self.bkg_model is not None:
            self.bkg_model.pretrain_siren()

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
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
            cur_epoch: current epoch, for training purpose only. By default 0.
            total_epoch: total num of epoch, for training purpose only. By default 300k.

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
        gpu_on_func = True if (self.is_cuda() and not rays_o.is_cuda) else False  # allow rays_o on cpu
        output = chunk_processing(
            self._forward, self.chunk_rays, gpu_on_func, flat_inputs, inference_only, get_progress, cur_epoch,
            total_epoch
        )
        for k, v in output.items():
            if isinstance(v, torch.Tensor) and batch_size * n_rays_per_batch == v.shape[0]:
                new_shape = tuple([batch_size, n_rays_per_batch] + list(v.shape)[1:])
                output[k] = v.view(new_shape)
            else:
                output[k] = v

        return output

    def _merge_bkg_sigma(self, inputs, sigma, radiance, zvals, inference_only):
        """merge sigma and get inputs for ray marching together. Only for sigma blending mode
        inference_only helps to sample different zvals during training.
        All inputs flatten in (B, x) dim
        """
        sigma_all, radiance_all, zvals_all = sigma, radiance, zvals
        if self.bkg_model is not None and self.bkg_blend == 'sigma':
            output_bkg = self.bkg_model._forward(inputs, inference_only=inference_only, get_progress=True)
            sigma_bkg = output_bkg['progress_sigma']  # (B, n_bkg(-1))
            radiance_bkg = output_bkg['progress_radiance']  # (B, n_bkg(-1))
            zvals_bkg = output_bkg['progress_zvals']  # (B, n_bkg(-1))
            sigma_all = torch.cat([sigma, sigma_bkg], 1)  # (B, n_fg + n_bkg(-1)), already sorted
            radiance_all = torch.cat([radiance, radiance_bkg], 1)  # (B, n_fg + n_bkg(-1), 3), already sorted
            zvals_all = torch.cat([zvals, zvals_bkg], 1)  # (B, n_fg + n_bkg(-1)), already sorted

        return sigma_all, radiance_all, zvals_all

    def _merge_bkg_rgb(self, inputs, output, inference_only):
        """ blend fg + bkg for rgb and depth. mask is still for foreground only. Only for rgb blending mode
        inference_only helps to sample different zvals during training.
        All inputs flatten in (B, x) dim
        """
        if self.bkg_model is not None and self.bkg_blend == 'rgb':
            output_bkg = self.bkg_model._forward(inputs, inference_only=inference_only)  # not need sigma
            bkg_lamba = output['trans_shift'][:, -1]  # (B,) prob that light passed through foreground field
            output['rgb'] = output['rgb'] + bkg_lamba[:, None] * output_bkg['rgb']
            output['depth'] = output['depth'] + bkg_lamba * output_bkg['depth']

        return output

    def _get_n_fg(self, sigma):
        """Get the num of foreground pts. sigma is (B, n_sample/n_total) """
        n_fg = sigma.shape[1]
        if self.bkg_model is not None and self.bkg_blend == 'rgb' and self.rays_cfgs['add_inf_z'] is False:
            n_fg -= 1

        return n_fg

    def _get_near_far_from_rays(self, inputs):
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
            inputs['rays_o'], inputs['rays_d'], bounds, self.rays_cfgs['near'], self.rays_cfgs['far'],
            self.rays_cfgs['bounding_radius']
        )

        return near, far

    def _forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        """
        All the tensor are in chunk. B is total num of rays by grouping different samples in batch
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
            output is a dict with following keys:
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

    @torch.no_grad()
    def forward_pts_dir(self, pts: torch.Tensor, view_dir: torch.Tensor = None):
        """This function forward pts and view dir directly, only for inference the geometry/color
        Assert you only have geo_net and radiance_net

        Args:
            pts: torch.tensor (N_pts, 3), pts in world coord
            view_dir: torch.tensor (N_pts, 3) view dir associate with each point. It can be normal or others.
                      If None, use (0, 0, 0) as the dir for each point.
        Returns:
            output is a dict with following keys:
                sigma/sdf: torch.tensor (N_pts), geometry value for each point
                rgb: torch.tensor (N_pts, 3), color for each point
        """
        gpu_on_func = True if (self.is_cuda() and not pts.is_cuda) else False

        if view_dir is None:
            rays_d = torch.zeros_like(pts, dtype=pts.dtype).to(pts.device)
        else:
            rays_d = normalize(view_dir)  # norm view dir

        sigma, rgb = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, gpu_on_func, self.geo_net, self.radiance_net, pts, rays_d
        )

        return sigma, rgb

    @staticmethod
    def _forward_pts_dir(
        geo_net,
        radiance_net,
        pts: torch.Tensor,
        rays_d: torch.Tensor = None,
    ):
        """Core forward function to forward. Use chunk progress to call it will save memory for feature.

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

    @torch.no_grad()
    def forward_pts(self, pts: torch.Tensor):
        """This function forward pts directly, only for inference the geometry
        Assert you only have geo_net and radiance_net

        Args:
            pts: torch.tensor (N_pts, 3), pts in world coord

        Returns:
            output is a dict with following keys:
                sigma/sdf: torch.tensor (N_pts), geometry value for each point
        """
        gpu_on_func = True if (self.is_cuda() and not pts.is_cuda) else False
        sigma, _ = chunk_processing(self.geo_net, self.chunk_pts, gpu_on_func, pts)

        return sigma[..., 0]
