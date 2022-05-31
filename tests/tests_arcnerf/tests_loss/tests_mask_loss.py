#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.loss.mask_loss import MaskLoss
from common.utils.cfgs_utils import dict_to_obj


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10
        cls.n_rays = 16
        cls.bn_tensor = torch.ones((cls.batch_size, cls.n_rays))

    def tests_maskloss(self):
        data = {'mask': self.bn_tensor.clone()}
        output = {'mask': self.bn_tensor.clone()}
        loss = MaskLoss(dict_to_obj({}))
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        loss = MaskLoss(dict_to_obj({'do_mean': False}))
        res = loss(data, output)
        self.assertEqual(res.shape, (self.batch_size, self.n_rays))
        l1loss = MaskLoss(dict_to_obj({'loss_type': 'L1'}))
        res = l1loss(data, output)
        self.assertEqual(res.shape, ())
        bce_loss = MaskLoss(dict_to_obj({'loss_type': 'BCE'}))
        res = bce_loss(data, output)
        self.assertEqual(res.shape, ())

    def tests_maskcfloss(self):
        data = {'mask': self.bn_tensor.clone()}
        output = {'mask_coarse': self.bn_tensor.clone()}
        loss = MaskLoss(dict_to_obj({'keys': ['mask_coarse']}))
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        output['mask_fine'] = self.bn_tensor.clone()
        loss = MaskLoss(dict_to_obj({'keys': ['mask_coarse', 'mask_fine']}))
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        l1loss = MaskLoss(dict_to_obj({'keys': ['mask_coarse', 'mask_fine'], 'loss_type': 'L1'}))
        res = l1loss(data, output)
        self.assertEqual(res.shape, ())
        bce_loss = MaskLoss(dict_to_obj({'keys': ['mask_coarse', 'mask_fine'], 'loss_type': 'BCE'}))
        res = bce_loss(data, output)
        self.assertEqual(res.shape, ())
