#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import torch

from . import log_model_info
from arcnerf.models import build_model
from common.utils.cfgs_utils import load_configs, obj_to_dict, dict_to_obj
from common.utils.logger import Logger

CONFIG_DIR = osp.abspath(osp.join(__file__, '../../../..', 'configs', 'models'))
RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 1
        cls.n_rays = 72 * 35

    @staticmethod
    def add_model_field(cfgs):
        new_cfgs = {'model': obj_to_dict(cfgs)}

        return dict_to_obj(new_cfgs)

    def _test_pts_dir_forward(self, model, pts, view_dir):
        sigma = model.forward_pts(pts)
        self.assertEqual(sigma.shape, (self.n_rays, ))
        sigma, rgb = model.forward_pts_dir(pts, view_dir)
        self.assertEqual(sigma.shape, (self.n_rays, ))
        self.assertEqual(rgb.shape, (self.n_rays, 3))

    def _test_forward(self, model, feed_in):
        output = model(feed_in)
        self.assertEqual(output['rgb'].shape, (self.batch_size, self.n_rays, 3))
        self.assertEqual(output['depth'].shape, (
            self.batch_size,
            self.n_rays,
        ))
        self.assertEqual(output['mask'].shape, (
            self.batch_size,
            self.n_rays,
        ))

    def tests_nerfpp_model(self):
        cfgs = load_configs(osp.join(osp.join(CONFIG_DIR, 'nerfpp.yaml')), None)
        cfgs = self.add_model_field(cfgs.model.background)
        logger = Logger(path=osp.join(RESULT_DIR, 'bkg_nerfpp.txt'), keep_console=False)
        model = build_model(cfgs, logger)

        feed_in = {
            'rays_o': torch.rand(self.batch_size, self.n_rays, 3),
            'rays_d': torch.rand(self.batch_size, self.n_rays, 3),
            'mask': torch.rand(self.batch_size, self.n_rays),
        }

        log_model_info(logger, model, feed_in, self.add_model_field(cfgs), self.batch_size, self.n_rays)

        # test forward
        self._test_forward(model, feed_in)

        # test_get_progress
        output = model(feed_in, get_progress=True)
        n_sample = cfgs.model.rays.n_sample
        gt_shape = (self.batch_size, self.n_rays, n_sample if cfgs.model.rays.add_inf_z else n_sample - 1)
        for key in ['sigma', 'zvals', 'alpha', 'trans_shift', 'weights']:
            self.assertEqual(output['progress_{}'.format(key)].shape, gt_shape)

        # direct inference
        pts = torch.ones(self.n_rays, 4)
        view_dir = torch.ones(self.n_rays, 3)
        self._test_pts_dir_forward(model, pts, view_dir)
