#!/usr/bin/python
# -*- coding: utf-8 -*-

from itertools import combinations
import os.path as osp
import unittest

import torch

from arcnerf.models.base_modules import FusedMLPGeoNet, FusedMLPRadianceNet
from common.utils.cfgs_utils import dict_to_obj
from common.utils.logger import Logger
from tests.tests_arcnerf.tests_models.tests_base_modules import log_base_model_info, RESULT_DIR


class TestDict(unittest.TestCase):
    """This tests the FusedMLPGeoNet and FusedMLPRadianceNet"""

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10

    def tests_fusemlp_geonet(self):
        if not torch.cuda.is_available():
            return

        x = torch.ones((self.batch_size, 3)).cuda()
        # normal case
        model = FusedMLPGeoNet(input_ch=3)
        y, feat = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        self.assertEqual(feat.shape, (self.batch_size, 128))
        # W_feat <= 0
        model = FusedMLPGeoNet(input_ch=3, W_feat=0)
        y, _ = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        # act
        cfg = {'type': 'softplus'}
        cfg = dict_to_obj(cfg)
        model = FusedMLPGeoNet(input_ch=3, act_cfg=cfg)
        y, feat = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        self.assertEqual(feat.shape, (self.batch_size, 128))
        # output act
        cfg = {'type': 'truncexp'}
        cfg = dict_to_obj(cfg)
        model = FusedMLPGeoNet(input_ch=3, out_act_cfg=cfg)
        y, feat = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        self.assertEqual(feat.shape, (self.batch_size, 128))
        # forward with normal output and geo value only
        model = FusedMLPGeoNet(input_ch=3)
        geo_value, feat, grad = model.forward_with_grad(x)
        self.assertEqual(x.shape, grad.shape)
        self.assertEqual(feat.shape, (self.batch_size, 128))
        geo_value = model.forward_geo_value(x)
        self.assertEqual(geo_value.shape, (self.batch_size, ))

    def tests_fusemlp_geonet_detail(self):
        if not torch.cuda.is_available():
            return

        model = FusedMLPGeoNet()
        logger = Logger(path=osp.join(RESULT_DIR, 'fusemlp_geonet.txt'), keep_console=False)
        n_pts = 4096 * 128
        feed_in = torch.ones((n_pts, 3)).cuda()
        log_base_model_info(logger, model, feed_in, n_pts)

    def tests_radiancenet(self):
        if not torch.cuda.is_available():
            return

        xyz = torch.rand((self.batch_size, 3)).cuda()
        view_dirs = torch.rand((self.batch_size, 3)).cuda()
        normals = torch.rand((self.batch_size, 3)).cuda()
        feat = torch.rand((self.batch_size, 128)).cuda()
        modes = ['p', 'v', 'n', 'f']
        modes = sum([list(map(list, combinations(modes, i))) for i in range(len(modes) + 1)], [])
        for mode in modes:
            if len(mode) == 0:
                continue
            mode = ''.join(mode)
            model = FusedMLPRadianceNet(mode=mode, W=128, D=8, W_feat_in=128)
            y = model(xyz, view_dirs, normals, feat)
            self.assertEqual(y.shape, (self.batch_size, 3))
            self.assertTrue(torch.all(torch.logical_and(y >= 0, y <= 1)))
