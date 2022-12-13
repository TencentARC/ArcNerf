# -*- coding: utf-8 -*-

import torch

from common.models.base_model import BaseModel
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing
from .dense_grid import DenseGrid
from .network import NGPNetwork
from .renderer import Renderer
from .sampler import DenseGridSampler


@MODEL_REGISTRY.register()
class NGP(BaseModel):
    """ NGP model. It mainly contains the operation of sampler, modeling, rendering.
        Ref: https://github.com/NVlabs/instant-ngp and lots of implementation
    """

    def __init__(self, cfgs):
        super(NGP, self).__init__(cfgs)
        # ray_cfgs
        self.ray_cfgs = self.read_ray_cfgs()
        self.chunk_rays = self.cfgs.model.chunk_rays
        self.chunk_pts = self.cfgs.model.chunk_pts

        # sampler
        self.sampler = DenseGridSampler(cfgs.model.sampler)
        # dense grid for volume record
        self.dense_grid = DenseGrid(cfgs.model.dense_grid, self.chunk_pts, self.get_ray_cfgs('n_sample'))
        # network
        self.net = NGPNetwork(cfgs.model.network, self.dense_grid.get_aabb_range())
        # renderer for rgb calculation
        self.renderer = Renderer(cfgs.model.renderer)

        self.aabb_range = self.dense_grid.get_aabb_range()

        # used for update batchsize
        self.max_allowance = (1 << self.get_ray_cfgs('log_max_allowance'))  # 4096 * 64
        self.measured_batch_size = 0
        self.measured_count = 0

    def read_ray_cfgs(self):
        """Read cfgs for ray, common case"""
        ray_cfgs = {
            'n_sample': get_value_from_cfgs_field(self.cfgs.model.rays, 'n_sample', 1024),
            'log_max_allowance': get_value_from_cfgs_field(self.cfgs.model.rays, 'log_max_allowance', 18)
        }
        return ray_cfgs

    def get_ray_cfgs(self, key=None):
        """Get ray cfgs by optional key"""
        if key is None:
            return self.ray_cfgs

        return self.ray_cfgs[key]

    @staticmethod
    def prepare_flatten_inputs(inputs, inference_only):
        """Prepare the inputs by flatten them from (B, N, ...) to (BN, ...)

        Args:
            inputs: a dict of torch tensor:
                inputs['img']: torch.tensor (B, N, 3), rgb images
                inputs['rays_o']: torch.tensor (B, N, 3), cam_loc/ray_start position
                inputs['rays_d']: torch.tensor (B, N, 3), rays direction
                inputs['mask']: torch.tensor (B, N), mask value in {0, 1}. optional
                inputs['bkg_color']: torch.tensor (B, N, 3), or None, random/fix bkg color
            inference_only: mode

        Returns:
            flatten_inputs:
                value in inputs flatten into (BN, ...)
        """
        batch_size, n_rays_per_batch = inputs['rays_o'].shape[:2]

        flat_inputs = {
            'img': inputs['img'].view(-1, 3),  # (BN, 3)
            'rays_o': inputs['rays_o'].view(-1, 3),  # (BN, 3)
            'rays_d': inputs['rays_d'].view(-1, 3),  # (BN, 3)
            'mask': inputs['mask'].view(-1) if 'mask' in inputs.keys() else None,  # (BN,)
            'bkg_color': inputs['bkg_color'].view(-1, 3) if 'bkg_color' in inputs.keys() else None,  # (BN, 3)
        }

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

    def forward(self, inputs, cur_epoch=0, inference_only=False):
        # optimize the dense grid
        if not inference_only and cur_epoch % self.dense_grid.epoch_optim == 0:
            self.optimize(cur_epoch)

        # prepare flatten inputs
        flat_inputs, batch_size, n_rays_per_batch = self.prepare_flatten_inputs(inputs, inference_only)

        # get output
        output = chunk_processing(self._forward, self.chunk_rays, False, flat_inputs, cur_epoch, inference_only)

        # reshape values from (B*N, ...) to (B, N, ...)
        output = self.reshape_output(output, batch_size, n_rays_per_batch)

        return output

    def _forward(self, inputs, cur_epoch=0, inference_only=False):
        """Real func to run the function"""
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        bkg_color = inputs['bkg_color']  # (B, 3) or None

        # get pts from sampler with bitfield
        bitfield = self.dense_grid.get_bitfield()
        n_grid, n_cascades = self.dense_grid.get_n_grid(), self.dense_grid.get_n_cascades()
        n_sample = self.get_ray_cfgs('n_sample')
        min_step, max_step = self.dense_grid.min_step_size(), self.dense_grid.max_step_size()
        # pts, dirs are not normalized
        pts, dirs, dt, numsteps_out, counter = self.sampler.sample(
            rays_o, rays_d, bitfield, n_sample, min_step, max_step, self.aabb_range, n_grid, n_cascades
        )

        # update dynamic batchsize
        if not inference_only:
            self.measured_count += 1.0
            self.measured_batch_size += float(self.max_allowance) / (float(counter) + 1)  # total pts per batch

        # run the model
        sigma, radiance = chunk_processing(self.net, self.chunk_pts, False, pts, dirs)

        # render
        rgb, alpha = self.renderer.render(
            sigma, radiance, numsteps_out, dt, bkg_color, inference_only
        )

        output = {'rgb': rgb, 'mask': alpha}

        return output

    def optimize(self, cur_epoch):
        self.dense_grid.update_density_grid(cur_epoch, self.net.forward_geo_value)

    def get_dense_grid(self):
        """get the dense grid"""
        return self.dense_grid

    def get_dynamic_factor(self):
        """return the dynamic factor to adjust batch size"""
        dynamic_factor = (self.measured_batch_size / self.measured_count)

        # reset for next round
        self.reset_measure()

        return dynamic_factor

    def reset_measure(self):
        """Reset the measure, in case validation call i[t"""
        self.measured_count = 0
        self.measured_batch_size = 0
