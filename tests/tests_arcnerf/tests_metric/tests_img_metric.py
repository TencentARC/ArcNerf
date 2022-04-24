#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.metric.img_metric import (PSNR, PSNRCoarse, PSNRFine, MaskPSNR, MaskPSNRCoarse, MaskPSNRFine)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10
        cls.n_rays = 16
        cls.bn_tensor = torch.ones((cls.batch_size, cls.n_rays))
        cls.bn3_tensor = torch.ones((cls.batch_size, cls.n_rays, 3))

    def tests_psnr(self):
        data = {'img': self.bn3_tensor.clone()}
        output = {'rgb': self.bn3_tensor.clone()}
        psnr = PSNR()
        metric = psnr(data, output)
        self.assertEqual(metric.shape, ())

    def tests_psnrcoarse(self):
        data = {'img': self.bn3_tensor.clone()}
        output = {'rgb_coarse': self.bn3_tensor.clone()}
        psnr = PSNRCoarse()
        metric = psnr(data, output)
        self.assertEqual(metric.shape, ())

    def tests_psnrfine(self):
        data = {'img': self.bn3_tensor.clone()}
        output = {'rgb_fine': self.bn3_tensor.clone()}
        psnr = PSNRFine()
        metric = psnr(data, output)
        self.assertEqual(metric.shape, ())

    def tests_maskpsnr(self):
        data = {'img': self.bn3_tensor.clone(), 'mask': self.bn_tensor.clone()}
        output = {'rgb': self.bn3_tensor.clone()}
        psnr = MaskPSNR()
        metric = psnr(data, output)
        self.assertEqual(metric.shape, ())

    def tests_maskpsnrcoarse(self):
        data = {'img': self.bn3_tensor.clone(), 'mask': self.bn_tensor.clone()}
        output = {'rgb_coarse': self.bn3_tensor.clone()}
        psnr = MaskPSNRCoarse()
        metric = psnr(data, output)
        self.assertEqual(metric.shape, ())

    def tests_maskpsnrfine(self):
        data = {'img': self.bn3_tensor.clone(), 'mask': self.bn_tensor.clone()}
        output = {'rgb_fine': self.bn3_tensor.clone()}
        psnr = MaskPSNRFine()
        metric = psnr(data, output)
        self.assertEqual(metric.shape, ())