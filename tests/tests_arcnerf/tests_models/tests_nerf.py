#!/usr/bin/python
# -*- coding: utf-8 -*-

import os.path as osp
import unittest

import torch

from arcnerf.models import build_model
from common.utils.cfgs_utils import load_configs

CONFIG = osp.abspath(osp.join(__file__, '../../../..', 'configs', 'models'))


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 2
        cls.n_rays = 32

    def setupNeRF(self):
        self.cfgs = load_configs(osp.join(CONFIG, 'nerf.yaml'), None)

    def setupNeRFFull(self):
        self.cfgs = load_configs(osp.join(CONFIG, 'nerf_full.yaml'), None)

    def tests_nerf_model(self):
        self.setupNeRF()
        model = build_model(self.cfgs, None)
        feed_in = {
            'img': torch.ones(self.batch_size, self.n_rays, 3),
            'mask': torch.ones(self.batch_size, self.n_rays),
            'rays_o': torch.rand(self.batch_size, self.n_rays, 3),
            'rays_d': torch.rand(self.batch_size, self.n_rays, 3),
            'bounds': torch.rand(self.batch_size, 2)
        }
        output = model(feed_in)
        self.assertEqual(output['rgb'].shape, (self.batch_size, self.n_rays, 3))
        self.assertEqual(output['depth'].shape, (self.batch_size, self.n_rays))
        self.assertEqual(output['mask'].shape, (self.batch_size, self.n_rays))

        # direct inference
        pts = torch.ones(self.n_rays, 3)
        view_dir = torch.ones(self.n_rays, 3)
        sigma, rgb = model.forward_pts_dir(pts, view_dir)
        self.assertEqual(sigma.shape, (self.n_rays, ))
        self.assertEqual(rgb.shape, (self.n_rays, 3))
