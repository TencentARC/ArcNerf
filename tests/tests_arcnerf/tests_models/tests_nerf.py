#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import torch

from . import log_model_info
from arcnerf.models import build_model
from common.utils.cfgs_utils import load_configs
from common.utils.logger import Logger

CONFIG = osp.abspath(osp.join(__file__, '../../../..', 'configs', 'models', 'nerf.yaml'))
RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 1
        cls.n_rays = 72 * 35
        cls.cfgs = load_configs(osp.join(CONFIG), None)
        cls.logger = Logger(path=osp.join(RESULT_DIR, 'nerf.txt'), keep_console=False)

    def tests_nerf_model(self):
        model = build_model(self.cfgs, None)
        feed_in = {
            'img': torch.ones(self.batch_size, self.n_rays, 3),
            'mask': torch.ones(self.batch_size, self.n_rays),
            'rays_o': torch.rand(self.batch_size, self.n_rays, 3),
            'rays_d': torch.rand(self.batch_size, self.n_rays, 3),
            'bounds': torch.rand(self.batch_size, 2)
        }

        log_model_info(self.logger, model, feed_in, self.cfgs, self.batch_size, self.n_rays)

        output = model(feed_in)
        self.assertEqual(output['rgb_coarse'].shape, (self.batch_size, self.n_rays, 3))
        self.assertEqual(output['depth_coarse'].shape, (self.batch_size, self.n_rays))
        self.assertEqual(output['mask_coarse'].shape, (self.batch_size, self.n_rays))

        if self.cfgs.model.rays.n_importance > 0:
            self.assertEqual(output['rgb_fine'].shape, (self.batch_size, self.n_rays, 3))
            self.assertEqual(output['depth_fine'].shape, (self.batch_size, self.n_rays))
            self.assertEqual(output['mask_fine'].shape, (self.batch_size, self.n_rays))

        # inference only
        output = model(feed_in, inference_only=True)
        self.assertTrue('rgb_coarse' not in output)
        self.assertTrue('depth_coarse' not in output)
        self.assertTrue('mask_coarse' not in output)

        # direct inference
        pts = torch.ones(self.n_rays, 3)
        view_dir = torch.ones(self.n_rays, 3)
        sigma, rgb = model.forward_pts_dir(pts, view_dir)
        self.assertEqual(sigma.shape, (self.n_rays, ))
        self.assertEqual(rgb.shape, (self.n_rays, 3))
