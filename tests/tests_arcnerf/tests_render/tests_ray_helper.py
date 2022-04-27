#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np
import torch

from arcnerf.render.ray_helper import (
    get_zvals_from_near_far, sample_pdf, sample_cdf, sample_ray_marching_output_by_index, ray_marching
)
from common.utils.torch_utils import np_wrapper, torch_to_np
from common.visual.plot_2d import draw_2d_components

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results', 'ray_helper'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_rays = 1
        cls.n_pts = 128
        cls.n_importance = 64
        cls.near = 2.0
        cls.far = 6.0

    def tests_get_zvals_from_near_far(self):
        near = np.ones((self.n_rays, 1)) * self.near
        far = np.ones((self.n_rays, 1)) * self.far
        # choices
        inclusive = [True, False]
        inverse_linear = [True, False]
        perturb = [True, False]
        # show lines
        legends = []
        points = []
        lines = []
        count = 1
        for i1 in inclusive:
            for i2 in inverse_linear:
                for p in perturb:
                    zvals = np_wrapper(get_zvals_from_near_far, near, far, self.n_pts, i1, i2, p)[0].tolist()
                    y = [count] * len(zvals)
                    points.append([zvals, y])
                    lines.append([zvals, y])
                    legends.append('Inclusive: {} - Inverse: {} - Perturb{}'.format(i1, i2, p))
                    count += 1

        file_path = osp.join(RESULT_DIR, 'get_zvals_from_near_far.png')
        draw_2d_components(
            points=points,
            lines=lines,
            legends=legends,
            xlabel='zvals',
            ylabel='',
            title='zvals by different method',
            save_path=file_path
        )

    def create_systhesis_ray_input(self):
        """Sample density distribution and test ray marching"""
        offset = int(self.n_pts / 3.0)
        level = 20
        # sigma
        sigma = [0.0 for _ in range(offset)]
        sigma.extend([(i + 1) * level for i in range(int(offset / 5.0) + 1)])
        sigma.extend([(int(offset / 5.0) + 1) * level for _ in range(3 * int((offset / 5.0) + 1))])
        sigma.extend([(int(offset / 5.0) + 1) * level - (i + 1) * level for i in range(int(offset / 5.0) + 1)])
        sigma.extend([0.0 for _ in range(offset)])
        sigma = np.array(sigma[:self.n_pts])[None, :]
        # zvals
        zvals = np.linspace(self.near, self.far, self.n_pts)[None, :]

        return sigma, zvals

    def tests_ray_marching(self):
        # all positive
        sigma, zvals = self.create_systhesis_ray_input()
        output = np_wrapper(ray_marching, sigma, None, zvals)
        self.assertIsNone(output['rgb'])
        self.assertEqual(output['depth'].shape, (1, ))
        self.assertEqual(output['mask'].shape, (1, ))
        self.assertEqual(output['weights'].shape, (1, self.n_pts - 1))
        self.assertEqual(output['zvals'].shape, (1, self.n_pts - 1))
        self.assertEqual(output['weights'].shape, (1, self.n_pts - 1))
        self.assertEqual(output['alpha'].shape, (1, self.n_pts - 1))
        self.assertEqual(output['trans_shift'].shape, (1, self.n_pts - 1))

        visual_list, _ = sample_ray_marching_output_by_index(output)
        visual_list = visual_list[0]

        file_path = osp.join(RESULT_DIR, 'ray_marching_all_pos.png')
        draw_2d_components(
            points=visual_list['points'],
            lines=visual_list['lines'],
            legends=visual_list['legends'],
            xlabel='zvals',
            ylabel='',
            title='ray marching from synthesis input(all positive value)',
            save_path=file_path
        )

        # all negative
        sigma, zvals = self.create_systhesis_ray_input()
        sigma -= (sigma.max(1) + 20.0)
        output = np_wrapper(ray_marching, sigma, None, zvals)

        visual_list, _ = sample_ray_marching_output_by_index(output)
        visual_list = visual_list[0]

        file_path = osp.join(RESULT_DIR, 'ray_marching_all_neg.png')
        draw_2d_components(
            points=visual_list['points'],
            lines=visual_list['lines'],
            legends=visual_list['legends'],
            xlabel='zvals',
            ylabel='',
            title='ray marching from synthesis input(all negative value)',
            save_path=file_path
        )

        # some negative
        sigma, zvals = self.create_systhesis_ray_input()
        sigma -= (sigma.max(1) / 2.0)
        output = np_wrapper(ray_marching, sigma, None, zvals)

        visual_list, _ = sample_ray_marching_output_by_index(output)
        visual_list = visual_list[0]

        file_path = osp.join(RESULT_DIR, 'ray_marching_pos_neg.png')
        draw_2d_components(
            points=visual_list['points'],
            lines=visual_list['lines'],
            legends=visual_list['legends'],
            xlabel='zvals',
            ylabel='',
            title='ray marching from synthesis input(both pos and neg value)',
            save_path=file_path
        )

    def tests_sample_pdf(self):
        sigma, zvals = self.create_systhesis_ray_input()
        output = np_wrapper(ray_marching, sigma, None, zvals)
        visual_list, _ = sample_ray_marching_output_by_index(output)
        visual_list = visual_list[0]
        pdf = output['weights']

        zvals_mid = 0.5 * (zvals[:, 1:] + zvals[:, :-1])
        _zvals = np_wrapper(sample_pdf, zvals_mid, pdf[:, :-2], self.n_importance)

        points = visual_list['points']
        points.append([_zvals[0], [-1.5] * _zvals.shape[-1]])

        file_path = osp.join(RESULT_DIR, 'sample_pdf.png')
        draw_2d_components(
            points=visual_list['points'],
            lines=visual_list['lines'],
            legends=visual_list['legends'],
            xlabel='zvals',
            ylabel='',
            title='sample pdf from synthesis inputs',
            save_path=file_path
        )

    def tests_sample_cdf_detail(self):
        sigma, zvals = self.create_systhesis_ray_input()  # (1, n_pts)
        start_idx = int(self.n_pts / 3.5)
        end_idx = int(1.5 * start_idx)
        sigma = sigma[:, start_idx:end_idx]
        zvals = zvals[:, start_idx:end_idx]
        output = np_wrapper(ray_marching, sigma, None, zvals, True, 0.0, False)
        weights = torch_to_np(output['weights'])[:, :-1]  # (1, n_pts-1)
        pdf = weights / np.sum(weights, axis=-1, keepdims=True)  # (1, n_pts-1)
        cdf = np_wrapper(torch.cumsum, pdf, -1)  # (1, n_pts-1)
        weights = np.concatenate([np.zeros_like(weights[:, :1]), weights], -1)  # (1, n_pts)
        pdf = np.concatenate([np.zeros_like(pdf[:, :1]), pdf], -1)  # (1, n_pts)
        cdf = np.concatenate([np.zeros_like(cdf[:, :1]), cdf], -1)  # (1, n_pts)
        samples = np_wrapper(sample_cdf, zvals, cdf, 10, False)

        lines = [[zvals[0], weights[0]], [zvals[0], pdf[0]], [zvals[0], cdf[0]]]
        legends = ['weights', 'pdf', 'cdf']
        points = [[zvals[0], [-0.1] * len(zvals[0])], [samples[0], [0] * len(samples[0])]]

        file_path = osp.join(RESULT_DIR, 'sample_cdf.png')
        draw_2d_components(
            points=points,
            lines=lines,
            legends=legends,
            xlabel='zvals',
            ylabel='',
            title='sample 10 pts from cdf from synthesis inputs',
            save_path=file_path
        )
