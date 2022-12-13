# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from common.utils.cfgs_utils import get_value_from_cfgs_field
from simplengp.ops import rays_sampler


class DenseGridSampler(nn.Module):
    """A dense grid sampler """

    def __init__(self, cfgs):
        super(DenseGridSampler, self).__init__()

        # cfgs
        self.near_distance = get_value_from_cfgs_field(cfgs, 'near_distance', 0.2)
        self.cone_angle = get_value_from_cfgs_field(cfgs, 'cone_angle', 0.0)  # this controls the sampling schedule

    def sample(self, rays_o, rays_d, bitfield, n_sample, min_step, max_step, aabb_range, n_grid, n_cascades):
        """Sample points from sparse bitfield"""
        return rays_sampler(
            rays_o, rays_d, bitfield, self.near_distance, n_sample,
            min_step, max_step, self.cone_angle, aabb_range, n_grid, n_cascades
        )

    @staticmethod
    def get_num_empty_rays(numsteps_out):
        """Num of rays that do not sample any pts"""
        return int((numsteps_out[:, 0] == 0).sum())

    @staticmethod
    def get_avg_sample_num(numsteps_out):
        """Avg sample num on each rays, include empty rays"""
        return (numsteps_out[:, 0]).type(torch.float32).mean()

    @staticmethod
    def get_avg_sample_num_in_sampled_rays(numsteps_out):
        """Avg sample num on each rays, do not include empty rays"""
        n_samples = numsteps_out[:, 0]

        return (n_samples[n_samples != 0]).type(torch.float32).mean()
