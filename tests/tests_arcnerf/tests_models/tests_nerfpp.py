#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp

from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.visual.plot_3d import draw_3d_components
from tests.tests_arcnerf.tests_models import TestModelDict
from common.utils.torch_utils import torch_to_np

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results', 'bkg_model'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestNerfPPDict(TestModelDict):

    def tests_model(self):
        cfgs, logger = self.get_cfgs_logger('nerfpp.yaml', 'nerfpp.txt')
        model = self.build_model_to_cuda(cfgs, logger)

        feed_in = self.create_feed_in_to_cuda()
        self.log_model_info(logger, model, feed_in, cfgs)

        # without obj_bound structure
        self.run_model_tests(model, feed_in, cfgs)

        # add volume and test
        model = self.add_volume_structure_to_fg_model(model)
        self.run_model_tests(model, feed_in, cfgs)

        # add sphere and test
        model = self.add_sphere_structure_to_fg_model(model)
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
        # visual plot the sampling in bkg
        rays_o = feed_in['rays_o'][0, :1, :]  # keep 1 rays
        rays_d = feed_in['rays_d'][0, :1, :]
        zvals, radius = model.get_bkg_model().get_zvals_outside_sphere(
            rays_o, rays_d, inference_only=True
        )  # (1, n_pts, 1)
        file_path = osp.join(RESULT_DIR, 'nerfpp_outside_sample.png')
        # keep some sphere for visual only
        max_sample = 16
        radius = radius[0, :, 0].tolist()[:max_sample]
        pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)[0][:max_sample]
        draw_3d_components(
            points=torch_to_np(pts),
            rays=[torch_to_np(rays_o[:max_sample]), torch_to_np(rays_d[:max_sample])],
            ray_linewidth=5,
            sphere_radius=radius,
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )
