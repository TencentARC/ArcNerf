# -*- coding: utf-8 -*-

import torch

from common.models.base_model import BaseModel
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.torch_utils import chunk_processing


class Base3dModel(BaseModel):
    """Base model for 3d reconstruction, mainly for reading cfgs. """

    def __init__(self, cfgs):
        super(Base3dModel, self).__init__(cfgs)
        # ray_cfgs
        self.rays_cfgs = self.read_ray_cfgs()
        self.chunk_rays = self.cfgs.model.chunk_rays  # for n_rays together, do not consider n_pts on ray
        self.chunk_pts = self.cfgs.model.chunk_pts  # for n_pts together, only for model forward

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

    def get_chunk_rays(self):
        """Get the chunk rays num"""
        return self.chunk_rays

    def get_chunk_pts(self):
        """Get the chunk pts num"""
        return self.chunk_pts

    def forward(self, inputs, inference_only=False, get_progress=False):
        """The forward function actually call chunk process func _forward
        to avoid large memory at same time.
        Do not call this directly using chunk since the tensor are not flatten to represent batch of rays.

        Args:
            inputs['rays_o']: torch.tensor (B, N, 3), cam_loc/ray_start position
            inputs['rays_d']: torch.tensor (B, N, 3), view dir(assume normed)
            inputs['img']: torch.tensor (B, N, 3), rgb value in 0-1, optional
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
        img = None
        if 'img' in inputs:
            img = inputs['img'].view(-1, 3)  # (BN, 3)
        flat_inputs['img'] = img

        bounds = None
        if 'bounds' in inputs:
            bounds = inputs['bounds'].view(-1, 2)  # (BN, 3)
        flat_inputs['bounds'] = bounds

        masks = None
        if 'masks' in inputs:
            masks = inputs['masks'].view(-1)  # (BN,)
        flat_inputs['masks'] = masks

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
        """The core forward function, each process a chunk of rays with components in (B, x)"""
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
        raise NotImplementedError('Please implement the core forward_pts_dir function for simple extracting')

    @torch.no_grad()
    def forward_pts(self, pts: torch.Tensor):
        """This function forward pts directly, only for inference the geometry

        Args:
            pts: torch.tensor (N_pts, 3), pts in world coord

        Returns:
            output is a dict with following keys:
                sigma/sdf: torch.tensor (N_pts), geometry value for each point
        """
        raise NotImplementedError('Please implement the core forward_pts function for getting sigma or sdf')
