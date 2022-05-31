#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.loss.geo_loss import EikonalLoss
from common.utils.cfgs_utils import dict_to_obj


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
        loss = EikonalLoss(dict_to_obj({}))
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        loss = EikonalLoss(dict_to_obj({'do_mean': False}))
        res = loss(data, output)
        self.assertEqual(res.shape, (self.batch_size, self.n_rays))
        mask_loss = EikonalLoss(dict_to_obj({'use_mask': True}))
        res = mask_loss(data, output)
        self.assertEqual(res.shape, ())
        mask_loss = EikonalLoss(dict_to_obj({'use_mask': True, 'do_mean': False}))
        res = mask_loss(data, output)
        self.assertEqual(res.shape, (self.batch_size, self.n_rays))

    def tests_eikonalPTloss(self):
        data = {'mask': self.bn_tensor.clone()}
        output = {'normal_pts': self.bnp3_tensor.clone()}
        loss = EikonalLoss(dict_to_obj({'key': 'normal_pts'}))
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        loss = EikonalLoss(dict_to_obj({'key': 'normal_pts', 'do_mean': False}))
        res = loss(data, output)
        self.assertEqual(res.shape, (self.batch_size, self.n_rays, self.n_pts))
        mask_loss = EikonalLoss(dict_to_obj({'key': 'normal_pts', 'use_mask': True}))
        res = mask_loss(data, output)
        self.assertEqual(res.shape, ())
        mask_loss = EikonalLoss(dict_to_obj({'key': 'normal_pts', 'use_mask': True, 'do_mean': False}))
        res = mask_loss(data, output)
        self.assertEqual(res.shape, (self.batch_size, self.n_rays, self.n_pts))
