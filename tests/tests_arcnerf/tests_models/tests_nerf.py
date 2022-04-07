#!/usr/bin/python
# -*- coding: utf-8 -*-

import os.path as osp
import unittest

import torch

from arcnerf.models import build_model
from common.utils.cfgs_utils import load_configs

CONFIG = osp.abspath(osp.join(__file__, '../../../..', 'configs', 'models', 'nerf.yaml'))


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfgs = load_configs(CONFIG, None)
        cls.batch_size = 10

    def tests_nerf_model(self):
        self.model = build_model(self.cfgs, None)
        inputs = {'x': torch.ones((self.batch_size, 3)), 'view_dirs': torch.ones((self.batch_size, 3))}
        sigma, rgb = self.model(inputs)
        self.assertEqual(sigma.shape, (self.batch_size, 1))
        self.assertEqual(rgb.shape, (self.batch_size, 3))
