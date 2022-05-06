#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.loss.geo_loss import (EikonalLoss, EikonalMaskLoss, EikonalPTLoss, EikonalPTMaskLoss)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10
        cls.n_rays = 16
        cls.n_pts = 32
        cls.bn_tensor = torch.ones((cls.batch_size, cls.n_rays))
        cls.bnp_tensor = torch.ones((cls.batch_size, cls.n_rays, cls.n_pts))
        cls.bnp3_tensor = torch.ones((cls.batch_size, cls.n_rays, cls.n_pts, 3))

    def tests_eikonalloss(self):
        data = {'mask': self.bn_tensor.clone()}
        output = {'normal': self.bnp_tensor.clone()}
        loss = EikonalLoss()
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        mask_loss = EikonalMaskLoss()
        res = mask_loss(data, output)
        self.assertEqual(res.shape, ())

    def tests_eikonalPTloss(self):
        data = {'mask': self.bn_tensor.clone()}
        output = {'normal_pts': self.bnp3_tensor.clone()}
        loss = EikonalPTLoss()
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        mask_loss = EikonalPTMaskLoss()
        res = mask_loss(data, output)
        self.assertEqual(res.shape, ())
