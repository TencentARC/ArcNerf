#!/usr/bin/python
# -*- coding: utf-8 -*-

import os.path as osp
import unittest

import torch

from arcnerf.models.base_modules import VolGeoNet

from common.utils.logger import Logger

from tests.tests_arcnerf.tests_models.tests_base_modules import log_base_model_info, RESULT_DIR


class TestDict(unittest.TestCase):
    """This tests the VolGeoNet and VolRadianceNet"""

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10

    def tests_geo_volnet(self):
        x = torch.rand((self.batch_size, 3)) * 0.5
        # normal case
        model = VolGeoNet(geometric_init=False, n_grid=128, side=1.5)
        y, feat = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        self.assertEqual(feat.shape, (self.batch_size, 256))
        # forward with normal output and geo value only
        geo_value = model.forward_geo_value(x)
        self.assertEqual(geo_value.shape, (self.batch_size, ))
        geo_value, feat, grad = model.forward_with_grad(x)
        self.assertEqual(x.shape, grad.shape)
        self.assertEqual(feat.shape, (self.batch_size, 256))
        # W_feat <= 0
        model = VolGeoNet(geometric_init=False, n_grid=128, side=1.5, W_feat=0)
        y, _ = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))

        # with nn
        model = VolGeoNet(geometric_init=False, n_grid=128, side=1.5, use_nn=True)
        y, feat = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        self.assertEqual(feat.shape, (self.batch_size, 256))

        # geo_init
        model = VolGeoNet(geometric_init=True, n_grid=128, side=1.5, use_nn=True)
        y, feat = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        self.assertEqual(feat.shape, (self.batch_size, 256))

    def tests_geo_volnet_detail(self):
        n_grid = 16
        model = VolGeoNet(geometric_init=False, n_grid=n_grid, side=1.5)
        logger = Logger(path=osp.join(RESULT_DIR, 'geo_volnet.txt'), keep_console=False)
        n_pts = 4096 * 128
        feed_in = torch.rand((n_pts, 3)) * 0.5
        log_base_model_info(logger, model, feed_in, n_pts)
        logger.add_log('    Num grid {}'.format(n_grid))
