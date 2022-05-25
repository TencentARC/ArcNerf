#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch.nn as nn

from arcnerf.models.base_modules import get_activation, Sine
from common.utils.cfgs_utils import dict_to_obj


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10

    def tests_get_activation(self):
        types = ['relu', 'softplus', 'leakyrelu', 'sine', 'sigmoid']
        act_types = [nn.ReLU, nn.Softplus, nn.LeakyReLU, Sine, nn.Sigmoid]
        for i, act_type in enumerate(types):
            cfg = {'type': act_type, 'slope': 1e-2, 'w': 30, 'beta': 100}
            cfg = dict_to_obj(cfg)
            act = get_activation(cfg)
            self.assertIsInstance(act, act_types[i])
