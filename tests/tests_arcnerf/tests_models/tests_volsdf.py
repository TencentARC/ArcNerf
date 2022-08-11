#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import torch

from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from arcnerf.models.volsdf_model import sdf_to_sigma
from arcnerf.render.ray_helper import (
    get_zvals_from_near_far, make_sample_rays, ray_marching, sample_ray_marching_output_by_index
)
from common.utils.torch_utils import np_wrapper, torch_to_np
from common.visual.plot_2d import draw_2d_components
from tests.tests_arcnerf.tests_models import TestModelDict, RESULT_DIR


class TestVolsdfDict(TestModelDict):

    def make_result_dir(self):
        self.result_dir = osp.join(RESULT_DIR, 'volsdf')
        os.makedirs(self.result_dir, exist_ok=True)

    def tests_model(self):
        cfgs, logger = self.get_cfgs_logger('volsdf.yaml', 'volsdf.txt')
        model = self.build_model_to_cuda(cfgs, logger)

        feed_in = self.create_feed_in_to_cuda()
        self.log_model_info(logger, model, feed_in, cfgs)

        # test forward
        self._test_forward(model, feed_in, extra_keys=['normal'], extra_bn3=[True])

        # test params
        self._test_forward_params_in(model, feed_in, ['beta'])

        # inference only
        self._test_forward_inference_only(model, feed_in)

        # test sample size
        n_sample = cfgs.model.rays.n_sample
        n_importance = cfgs.model.rays.n_importance
        n_total = n_sample + n_importance
        zvals = get_zvals_from_near_far(
            feed_in['near'].view(-1, 1), feed_in['far'].view(-1, 1), cfgs.model.rays.n_eval
        )  # (BN, 3)
        zvals, zvals_surface = model.get_fg_model().upsample_zvals(
            feed_in['rays_o'].view(-1, 3), feed_in['rays_d'].view(-1, 3), zvals, False,
            model.get_fg_model().forward_pts
        )
        self.assertEqual(zvals.shape, (self.batch_size * self.n_rays, n_total))
        self.assertEqual(zvals_surface.shape, (self.batch_size * self.n_rays, 1))

        # get progress
        progress_shape = (self.batch_size, self.n_rays, n_total - 1)
        self._test_forward_progress(model, feed_in, progress_shape)

        # direct inference
        pts, view_dir = self.create_pts_dir_to_cuda()
        self._test_pts_dir_forward(model, pts, view_dir)

        # opacity
        self._test_get_est_opacity(model, pts)

        # test sdf_to_sigma
        self._test_sdf_to_sigma()

        # test sample position
        self._test_sampling(model, cfgs)

    def _test_forward_inference_only(self, model, feed_in):
        """Test that all keys are not started with progress_"""
        output = model(feed_in, inference_only=True)
        self.assertEqual(output['rgb'].shape, (self.batch_size, self.n_rays, 3))
        self.assertTrue('rgb_coarse' not in output.keys())
        self.assertTrue('rgb_fine' not in output.keys())
        self.assertTrue(all([not k.startswith('progress_') for k in output.keys()]))
        self.assertTrue('normal_pts' not in output.keys())
        self.assertTrue('params' not in output.keys())

    def _test_sdf_to_sigma(self):
        beta = [1e-2, 1e-1, 1]
        sample_sdf = make_sample_rays(n_pts=128)

        self.make_result_dir()

        for b in beta:
            sigma = np_wrapper(sdf_to_sigma, sample_sdf['vals'], b)
            output = np_wrapper(ray_marching, sigma, None, sample_sdf['zvals'])
            visual_list, _ = sample_ray_marching_output_by_index(output)
            visual_list = visual_list[0]

            # add sdf
            visual_list['lines'].append([sample_sdf['zvals_list'], sample_sdf['vals_list']])
            visual_list['legends'].append('sdf')

            file_path = osp.join(self.result_dir, 'volsdf_sdf_to_sigma{}.png'.format(b))
            draw_2d_components(
                points=visual_list['points'],
                lines=visual_list['lines'],
                legends=visual_list['legends'],
                xlabel='zvals',
                ylabel='',
                title='ray marching from volsdf sdf_to_sigma. Beta {}'.format(str(b)),
                save_path=file_path
            )

    def _test_sampling(self, model, cfgs):

        def sdf_func(pts: torch.Tensor):
            """pts: (B, 3). sdf (B)"""
            radius = 1.0
            return torch.norm(pts, dim=1) - radius

        def get_2d_output(zvals, zvals_sample, zvals_surface, sdf):
            """Return a dict for 2d """
            # write output
            res = {'points': [], 'lines': [], 'legends': []}
            # all coarse sample
            x = torch_to_np(zvals[0]).tolist()
            res['points'].append([x, [-1] * len(x)])
            # real sampling
            x = torch_to_np(zvals_sample[0]).tolist()
            res['points'].append([x, [-1.5] * len(x)])
            # surface point
            x = torch_to_np(zvals_surface[0]).tolist()
            res['points'].append([x, [-2] * len(x)])

            # sdf line with coarse zvals
            res['lines'].append([torch_to_np(zvals[0]).tolist(), sdf.tolist()])
            res['legends'].append('sdf')

            res['lines'].append([torch_to_np(zvals[0]).tolist(), [0] * zvals.shape[1]])
            res['legends'].append('surface')

            return res

        self.make_result_dir()

        # ray
        rays_o = torch.tensor([1.5, 1.5, 1.5])[None]  # (1, 3)
        rays_d = torch.tensor([-1.0, -1.0, -1.0])[None]  # (1, 3)
        rays_d = normalize(rays_d)

        # near/far zvals of coarse sampling
        near = torch.tensor([0.5])[None]
        far = torch.tensor([5.5])[None]
        zvals = get_zvals_from_near_far(near, far, cfgs.model.rays.n_eval)  # (1, N_eval)
        pts = get_ray_points_by_zvals(rays_o, rays_d, zvals).view(-1, 3)  # (N_eval, 3)
        sdf = sdf_func(pts)  # (N_eval)

        # put to gpu
        if torch.cuda.is_available():
            rays_o = rays_o.cuda()
            rays_d = rays_d.cuda()
            zvals = zvals.cuda()

        # with n_importance
        zvals_sample, zvals_surface = model.get_fg_model().upsample_zvals(
            rays_o, rays_d, zvals, True, sdf_func
        )  # (1, n_importance+n_sample), (1, 1)

        res = get_2d_output(zvals, zvals_sample, zvals_surface, sdf)

        file_path = osp.join(
            self.result_dir, 'volsdf_sample_zvals_nimportance{}.png'.format(cfgs.model.rays.n_importance)
        )
        draw_2d_components(
            points=res['points'],
            lines=res['lines'],
            legends=res['legends'],
            title='Sample with n_importance',
            save_path=file_path
        )

        # with no n_importance(all near surface)
        model.get_fg_model().set_ray_cfgs('n_importance', 0)

        zvals_sample, zvals_surface = model.get_fg_model().upsample_zvals(
            rays_o, rays_d, zvals, True, sdf_func
        )  # (1, n_sample), (1, 1)

        res = get_2d_output(zvals, zvals_sample, zvals_surface, sdf)

        file_path = osp.join(self.result_dir, 'volsdf_sample_zvals_nimportance0.png')

        draw_2d_components(
            points=res['points'],
            lines=res['lines'],
            legends=res['legends'],
            title='Sample with no n_importance',
            save_path=file_path
        )
