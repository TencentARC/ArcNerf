#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.loss.img_loss import ImgLoss
from common.utils.cfgs_utils import dict_to_obj


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10
        cls.n_rays = 16
        cls.bn_tensor = torch.ones((cls.batch_size, cls.n_rays))
        cls.bn3_tensor = torch.ones((cls.batch_size, cls.n_rays, 3))

    def tests_imgloss(self):
        data = {'img': self.bn3_tensor.clone()}
        output = {'rgb': self.bn3_tensor.clone()}
        loss = ImgLoss(dict_to_obj({}))
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        loss = ImgLoss(dict_to_obj({'do_mean': False}))
        res = loss(data, output)
        self.assertEqual(res.shape, (self.batch_size, self.n_rays, 3))
        l1loss = ImgLoss(dict_to_obj({'loss_type': 'L1'}))
        res = l1loss(data, output)
        self.assertEqual(res.shape, ())
        huberloss = ImgLoss(dict_to_obj({'loss_type': 'Huber', 'delta': 0.1}))
        res = huberloss(data, output)
        self.assertEqual(res.shape, ())

    def tests_imgcfloss(self):
        data = {'img': self.bn3_tensor.clone()}
        output = {'rgb_coarse': self.bn3_tensor.clone()}
        loss = ImgLoss(dict_to_obj({'keys': ['rgb_coarse']}))
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        output['rgb_fine'] = self.bn3_tensor.clone()
        loss = ImgLoss(dict_to_obj({'keys': ['rgb_coarse', 'rgb_fine']}))
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        l1loss = ImgLoss(dict_to_obj({'keys': ['rgb_coarse', 'rgb_fine'], 'loss_type': 'L1'}))
        res = l1loss(data, output)
        self.assertEqual(res.shape, ())

    def tests_imgmaskloss(self):
        data = {'img': self.bn3_tensor.clone(), 'mask': self.bn_tensor.clone()}
        output = {'rgb': self.bn3_tensor.clone()}
        loss = ImgLoss(dict_to_obj({'use_mask': True}))
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        l1loss = ImgLoss(dict_to_obj({'use_mask': True, 'loss_type': 'L1'}))
        res = l1loss(data, output)
        self.assertEqual(res.shape, ())

    def tests_imgcfmaskloss(self):
        data = {'img': self.bn3_tensor.clone(), 'mask': self.bn_tensor.clone()}
        output = {'rgb_coarse': self.bn3_tensor.clone()}
        loss = ImgLoss(dict_to_obj({'keys': ['rgb_coarse'], 'use_mask': True}))
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        output['rgb_fine'] = self.bn3_tensor.clone()
        loss = ImgLoss(dict_to_obj({'keys': ['rgb_coarse', 'rgb_fine'], 'use_mask': True}))
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        l1loss = ImgLoss(dict_to_obj({'keys': ['rgb_coarse', 'rgb_fine'], 'use_mask': True, 'loss_type': 'L1'}))
        res = l1loss(data, output)
        self.assertEqual(res.shape, ())
