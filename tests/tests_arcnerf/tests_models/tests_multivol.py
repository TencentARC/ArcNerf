#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp

import numpy as np
import torch

from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.volume import Volume
from arcnerf.visual.plot_3d import draw_3d_components
from tests.tests_arcnerf.tests_models import TestModelDict
from common.utils.torch_utils import torch_to_np

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results', 'bkg_model'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestNerfPPDict(TestModelDict):

    def tests_model(self):
        cfgs, logger = self.get_cfgs_logger('nerf_multivol.yaml', 'nerf_multivol.txt')
        model = self.build_model_to_cuda(cfgs, logger)

        # try optimize
        model.optimize(16)

        feed_in = self.create_feed_in_to_cuda()
        self.log_model_info(logger, model, feed_in, cfgs)

        # without obj_bound structure
        self.run_model_tests(model, feed_in, cfgs)

    def run_model_tests(self, model, feed_in, cfgs):
        # visual plot the samples
        self.plot_bkg_sample_visual(model, feed_in)

        # test forward
        self._test_forward(model, feed_in, '_coarse')

        if cfgs.model.rays.n_importance > 0:
            self._test_forward(model, feed_in, '_fine')

        # inference only
        self._test_forward_inference_only(model, feed_in)

        # direct pts/view
        pts, view_dir = self.create_pts_dir_to_cuda()
        self._test_pts_dir_forward(model, pts, view_dir)

        # opacity
        self._test_get_est_opacity(model, pts)

        # surface render
        self._test_surface_render(model, feed_in, method='secant_root_finding', grad_dir='descent')

    @staticmethod
    def plot_bkg_sample_visual(model, feed_in):
        """Plot the visual for outside sampling"""
        # visual plot the sampling in bkg volume
        n_rays = 512
        rays_o = feed_in['rays_o'][0, :n_rays]
        rays_d = feed_in['rays_d'][0, :n_rays]
        bkg_model = model.get_bkg_model()
        # add pruning
        bkg_model.density_bitfield = torch.randint(
            low=0, high=255, size=bkg_model.density_bitfield.shape
        ).type(torch.uint8).to(rays_o.device)

        # sample pts
        near, far = bkg_model.get_near_far_from_rays(rays_o, rays_d)
        n_pts = 1024
        zvals, mask_pts = bkg_model.get_zvals_from_near_far(near, far, n_pts, rays_o, rays_d)
        pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (N_rays, N_pts, 3)
        valid_pts = pts[mask_pts]

        # draw multi-res volume
        basic_volume = bkg_model.basic_volume
        origin = basic_volume.get_origin()  # (3, )
        n_cascade = bkg_model.n_cascade
        max_len = [[x * 2**c for x in basic_volume.get_len()] for c in range(1, n_cascade)]
        volumes = [Volume(origin=origin, xyz_len=max_len) for max_len in max_len]

        vol_dict = {
            'grid_pts': torch_to_np(basic_volume.get_corner()),  # (8, 3)
            'lines': basic_volume.get_bound_lines(),  # (2*6, 3)
            # 'faces': basic_volume.get_bound_faces()  # (3(n+1)n^2, 4, 3)
        }
        for v in volumes:
            vol_dict = {
                'grid_pts': np.concatenate([vol_dict['grid_pts'], torch_to_np(v.get_corner())], axis=0),
                'lines': np.concatenate([vol_dict['lines'], v.get_bound_lines()], axis=0),
                # 'faces': np.concatenate([vol_dict['faces'], v.get_bound_faces()], axis=0)
            }

        file_path = osp.join(RESULT_DIR, 'multivol_outside_sample.png')
        draw_3d_components(
            points=torch_to_np(valid_pts),
            point_size=10,
            rays=[torch_to_np(rays_o), torch_to_np(rays_d)],
            volume=vol_dict,
            title='Sampling in multi-res vol, the inner vol should be skipped',
            save_path=file_path,
            plotly=True,
            plotly_html=True,
        )
