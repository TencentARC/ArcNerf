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
        self.chunk_size = self.cfgs.model.chunk_size  # for n_rays together, do not consider n_pts on ray

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

    def forward(self, inputs, inference_only=False, get_progress=False):
        """The forward function actually call chunk process func _forward
        to avoid large memory at same time.
        Do not call this directly using chunk since the tensor are not flatten to represent batch of rays.

        Args:
            inputs['img']: torch.tensor (B, N, 3), rgb value in 0-1
            inputs['rays_o']: torch.tensor (B, N, 3), cam_loc/ray_start position
            inputs['rays_d']: torch.tensor (B, N, 3), view dir(assume normed)
            inputs['mask']: torch.tensor (B, N), mask value in {0, 1}. optional
            inputs['bound']: torch.tensor (B, 2). optional
            inference_only: If True, will not output coarse results. By default False
            get_progress: If True, output some progress for recording, can not used in inference only mode.
                          By default False

        Returns:
            output is a dict keys like (rgb, rgb_coarse, rgb_dense, depth, etc) based on the _forward function.
        """
        flat_inputs = {}
        img = inputs['img'].view(-1, 3)  # (BN, 3)
        rays_o = inputs['rays_o'].view(-1, 3)  # (BN, 3)
        rays_d = inputs['rays_d'].view(-1, 3)  # (BN, 3)
        batch_size, n_rays_per_batch = inputs['rays_o'].shape[:2]
        flat_inputs['img'] = img
        flat_inputs['rays_o'] = rays_o
        flat_inputs['rays_d'] = rays_d

        bounds = None
        if 'bounds' in inputs:
            bounds = torch.repeat_interleave(inputs['bounds'], n_rays_per_batch, dim=0)  # (BN, 3)
        flat_inputs['bounds'] = bounds

        masks = None
        if 'masks' in inputs:
            masks = inputs['masks'].view(-1)  # (BN,)
        flat_inputs['masks'] = masks

        # all output tensor in (B*N, ...), reshape to (B, N, ...)
        output = chunk_processing(self._forward, self.chunk_size, flat_inputs, inference_only, get_progress)
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                old_shape = v.shape
                assert batch_size * n_rays_per_batch == old_shape[0], 'Invalid output shape...Not flatten...'
                new_shape = tuple([batch_size, n_rays_per_batch] + list(old_shape)[1:])
                output[k] = v.view(new_shape)

        return output

    def _forward(self, inputs, inference_only=False, get_progress=False):
        """The core forward function, each process a chunk of rays with components in (B, x)"""
        raise NotImplementedError('Please implement the core forward function')
