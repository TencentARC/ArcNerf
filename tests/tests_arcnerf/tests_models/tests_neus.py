#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp

import numpy as np

from arcnerf.models.neus_model import sdf_to_alpha, sdf_to_pdf
from arcnerf.render.ray_helper import make_sample_rays, ray_marching, sample_ray_marching_output_by_index
from common.utils.torch_utils import np_wrapper
from common.visual.plot_2d import draw_2d_components
from tests.tests_arcnerf.tests_models import TestModelDict, RESULT_DIR


class TestNeusDict(TestModelDict):

    def make_result_dir(self):
        self.result_dir = osp.join(RESULT_DIR, 'neus')
        os.makedirs(self.result_dir, exist_ok=True)

    def tests_model(self):
        cfgs, logger = self.get_cfgs_logger('neus.yaml', 'neus.txt')
        model = self.build_model_to_cuda(cfgs, logger)

        feed_in = self.create_feed_in_to_cuda()
        self.log_model_info(logger, model, feed_in, cfgs)

        # test forward
        self._test_forward(model, feed_in, extra_keys=['normal'], extra_bn3=[True])

        # test params
        self._test_forward_params_in(model, feed_in, ['scale'])

        # inference only
        self._test_forward_inference_only(model, feed_in)

        # get progress
        n_sample = cfgs.model.rays.n_sample
        n_importance = (cfgs.model.rays.n_importance // cfgs.model.rays.n_iter) * cfgs.model.rays.n_iter
        n_total = n_sample + n_importance
        progress_shape = (self.batch_size, self.n_rays, n_total - 1)
        self._test_forward_progress(model, feed_in, progress_shape)

        # direct inference
        pts, view_dir = self.create_pts_dir_to_cuda()
        self._test_pts_dir_forward(model, pts, view_dir)

        # test sdf_to_alpha
        self._test_sdf_to_alpha()

        # test weight
        self._test_sdf_weight()

    def _test_forward_inference_only(self, model, feed_in):
        """Test that all keys are not started with progress_"""
        output = model(feed_in, inference_only=True)
        self.assertEqual(output['rgb'].shape, (self.batch_size, self.n_rays, 3))
        self.assertTrue('rgb_coarse' not in output.keys())
        self.assertTrue('rgb_fine' not in output.keys())
        self.assertTrue(all([not k.startswith('progress_') for k in output.keys()]))
        self.assertTrue('normal_pts' not in output.keys())
        self.assertTrue('params' not in output.keys())

    def _test_sdf_to_alpha(self):
        """Test sdf to alpha under different scale"""
        scale = [1, 10, 100]
        sample_sdf = make_sample_rays(n_pts=128)

        self.make_result_dir()

        for s in scale:
            alpha = np_wrapper(sdf_to_alpha, sample_sdf['mid_vals'], sample_sdf['zvals'], sample_sdf['mid_slope'], s)
            output = np_wrapper(
                ray_marching, sample_sdf['mid_vals'], None, sample_sdf['mid_zvals'], False, 0.0, False, False, alpha
            )
            visual_list, _ = sample_ray_marching_output_by_index(output)

            # change sigma name to sdf
            visual_list = visual_list[0]
            visual_list['legends'][0] = 'sdf'

            file_path = osp.join(self.result_dir, 'neus_sdf_to_alpha_s{}.png'.format(s))
            draw_2d_components(
                points=visual_list['points'],
                lines=visual_list['lines'],
                legends=visual_list['legends'],
                xlabel='zvals',
                ylabel='',
                title='ray marching from neus sdf_to_alpha. Scale {}'.format(s),
                save_path=file_path
            )

    def _test_sdf_weight(self):
        """Test sdf to weight under different algorithm"""
        s = 100
        sample_sdf = make_sample_rays(n_pts=256)
        total_len = len(sample_sdf['zvals_list'])
        # only select a decreasing list
        start, end = int(total_len / 3.5), int(total_len / 2.5)
        sample_sdf['vals'] = sample_sdf['vals'][:, start:end]
        sample_sdf['mid_vals'] = sample_sdf['mid_vals'][:, start:end - 1]
        sample_sdf['zvals'] = sample_sdf['zvals'][:, start:end]
        sample_sdf['mid_zvals'] = sample_sdf['mid_zvals'][:, start:end - 1]
        sample_sdf['mid_slope'] = sample_sdf['mid_slope'][:, start:end - 1]

        self.make_result_dir()

        # use simple sdf to sigma
        sigma = np_wrapper(sdf_to_pdf, sample_sdf['vals'], s)
        output = np_wrapper(ray_marching, sigma, None, sample_sdf['zvals'], False, 0.0, False, False, None)
        visual_list, _ = sample_ray_marching_output_by_index(output)
        visual_list = visual_list[0]

        # use neus
        alpha = np_wrapper(sdf_to_alpha, sample_sdf['mid_vals'], sample_sdf['zvals'], sample_sdf['mid_slope'], s)
        output = np_wrapper(
            ray_marching, sample_sdf['mid_vals'], None, sample_sdf['mid_zvals'], False, 0.0, False, False, alpha
        )
        visual_list_neus, _ = sample_ray_marching_output_by_index(output)
        visual_list_neus = visual_list_neus[0]
        visual_list_neus['legends'][0] = 'sdf'

        # combine result
        points = visual_list_neus['points']
        lines = []
        legends = []

        for idx, legend in enumerate(visual_list_neus['legends']):
            if legend == 'sdf':
                lines.append(visual_list_neus['lines'][idx])
                legends.append('sdf')
            elif legend == 'weights':
                weights = visual_list_neus['lines'][idx]
                lines.append(weights)
                legends.append('neus_weights')
                # add a vertical weight for max
                max_weight_index = weights[1].index(max(weights[1]))
                max_zvals = weights[0][max_weight_index]
                lines.append([[max_zvals] * 30, np.linspace(-1, 2, 30)])
                legends.append('neus_max_weights')

        for idx, legend in enumerate(visual_list['legends']):
            if legend == 'weights':
                weights = visual_list['lines'][idx]
                lines.append(weights)
                legends.append('naive_weights')
                # add a vertical weight for max
                max_weight_index = weights[1].index(max(weights[1]))
                max_zvals = weights[0][max_weight_index]
                lines.append([[max_zvals] * 30, np.linspace(-1, 2, 30)])
                legends.append('naive_max_weights')

        file_path = osp.join(self.result_dir, 'neus_weight.png')
        draw_2d_components(
            points=points,
            lines=lines,
            legends=legends,
            xlabel='zvals',
            ylabel='',
            title='ray marching from neus by different method',
            save_path=file_path
        )
