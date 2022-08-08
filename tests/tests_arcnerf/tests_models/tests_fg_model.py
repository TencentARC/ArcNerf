#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import torch

from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from arcnerf.models.fg_model import FgModel
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.cfgs_utils import dict_to_obj
from common.utils.torch_utils import torch_to_np

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestModelDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_rays = 512
        cls.bounding_radius = 3.0
        cls.base_cfgs = {
            'model': {
                'chunk_rays': 4096,
                'chunk_pts': 4096 * 32,
                'rays': {
                    'bounding_radius': cls.bounding_radius
                }
            }
        }
        cls.result_dir = osp.join(RESULT_DIR, 'fg_model')
        os.makedirs(cls.result_dir, exist_ok=True)

    @staticmethod
    def to_cuda(item):
        """Move model or tensor to cuda"""
        if torch.cuda.is_available():
            item = item.cuda()

        return item

    def create_feed_in_to_cuda(self):
        rays_o = torch.rand(self.n_rays, 3) * 3.0
        rays_d = -normalize(rays_o)  # point to origin
        feed_in = {
            'rays_o': rays_o,
            'rays_d': rays_d,
        }

        for k, v in feed_in.items():
            feed_in[k] = self.to_cuda(v)

        return feed_in

    def run_fg_model_tests(self, model, type='none'):
        inputs = self.create_feed_in_to_cuda()
        model = self.to_cuda(model)
        # call get_near_far_from_rays
        near, far = model.get_near_far_from_rays(inputs)
        self.assertEqual(near.shape, (self.n_rays, 1))
        self.assertEqual(far.shape, (self.n_rays, 1))
        # draw the sampling pts
        zvals = model.get_zvals_from_near_far(near, far, 16)
        pts = get_ray_points_by_zvals(inputs['rays_o'], inputs['rays_d'], zvals)
        pts = torch_to_np(pts).reshape(-1, 3)

        file_path = osp.join(self.result_dir, 'struct_{}_sampling_pts.png'.format(type))
        draw_3d_components(
            points=pts,
            point_size=5.0,
            sphere_radius=self.bounding_radius,
            title='Sampling pts by inner obj bound - {}'.format(type),
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

    def tests_set_up_volume_bound_model(self):
        volume_model_cfgs = self.base_cfgs.copy()
        volume_cfgs = {'volume': {'side': 1.5}}
        volume_model_cfgs['model']['obj_bound'] = volume_cfgs
        fg_model = FgModel(dict_to_obj(volume_model_cfgs))
        self.run_fg_model_tests(fg_model, 'volume')

    def tests_set_up_sphere_bound_model(self):
        sphere_model_cfgs = self.base_cfgs.copy()
        sphere_cfgs = {'sphere': {'radius': 1.0}}
        sphere_model_cfgs['model']['obj_bound'] = sphere_cfgs
        fg_model = FgModel(dict_to_obj(sphere_model_cfgs))
        self.run_fg_model_tests(fg_model, 'sphere')

    def tests_set_up_no_bound_model(self):
        fg_model = FgModel(dict_to_obj(self.base_cfgs))
        self.run_fg_model_tests(fg_model)
