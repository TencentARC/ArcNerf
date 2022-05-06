#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.loss.img_loss import (
    ImgLoss, ImgL1Loss, ImgMaskLoss, ImgMaskL1Loss, ImgCFLoss, ImgCFL1Loss, ImgCFMaskLoss, ImgCFMaskL1Loss
)


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
        loss = ImgLoss()
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        l1loss = ImgL1Loss()
        res = l1loss(data, output)
        self.assertEqual(res.shape, ())

    def tests_imgcfloss(self):
        data = {'img': self.bn3_tensor.clone()}
        output = {'rgb_coarse': self.bn3_tensor.clone()}
        loss = ImgCFLoss()
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        output['rgb_fine'] = self.bn3_tensor.clone()
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        l1loss = ImgCFL1Loss()
        res = l1loss(data, output)
        self.assertEqual(res.shape, ())

    def tests_imgmaskloss(self):
        data = {'img': self.bn3_tensor.clone(), 'mask': self.bn_tensor.clone()}
        output = {'rgb': self.bn3_tensor.clone()}
        loss = ImgMaskLoss()
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        l1loss = ImgMaskL1Loss()
        res = l1loss(data, output)
        self.assertEqual(res.shape, ())

    def tests_imgcfmaskloss(self):
        data = {'img': self.bn3_tensor.clone(), 'mask': self.bn_tensor.clone()}
        output = {'rgb_coarse': self.bn3_tensor.clone()}
        loss = ImgCFMaskLoss()
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        output['rgb_fine'] = self.bn3_tensor.clone()
        res = loss(data, output)
        self.assertEqual(res.shape, ())
        l1loss = ImgCFMaskL1Loss()
        res = l1loss(data, output)
        self.assertEqual(res.shape, ())
