#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.loss.mask_loss import MaskLoss, MaskCFLoss


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10
        cls.n_rays = 16
        cls.bn_tensor = torch.ones((cls.batch_size, cls.n_rays))

    def tests_maskloss(self):
        data = {'mask': self.bn_tensor.clone()}
        output = {'mask': self.bn_tensor.clone()}
        loss = MaskLoss()
        res = loss(data, output)
        self.assertEqual(res.shape, ())

    def tests_maskcfloss(self):
        data = {'mask': self.bn_tensor.clone()}
        output = {'mask_coarse': self.bn_tensor.clone()}
        loss = MaskCFLoss()
        res = loss(data, output)
        self.assertEqual(res.shape, ())

        output['mask_fine'] = self.bn_tensor.clone()
        res = loss(data, output)
        self.assertEqual(res.shape, ())
