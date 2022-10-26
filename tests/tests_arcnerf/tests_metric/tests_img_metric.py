#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.metric.img_metric import PSNR, MaskPSNR, SSIM, MaskSSIM
from common.utils.cfgs_utils import dict_to_obj


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10
        cls.n_rays = 16
        cls.H = 8
        cls.W = 2
        cls.b_tensor = torch.ones((cls.batch_size))
        cls.bn_tensor = torch.ones((cls.batch_size, cls.n_rays))
        cls.bn3_tensor = torch.ones((cls.batch_size, cls.n_rays, 3))

    def tests_psnr(self):
        data = {'img': self.bn3_tensor.clone()}
        output = {'rgb': self.bn3_tensor.clone()}
        # full image psnr
        psnr = PSNR(dict_to_obj({}))
        metric = psnr(data, output)
        self.assertEqual(metric.shape, ())

    def tests_psnrcoarse(self):
        data = {'img': self.bn3_tensor.clone()}
        output = {'rgb_coarse': self.bn3_tensor.clone()}
        # full image psnr on coarse rgb
        psnr = PSNR(dict_to_obj({'key': 'rgb_coarse'}))
        metric = psnr(data, output)
        self.assertEqual(metric.shape, ())

    def tests_psnrfine(self):
        data = {'img': self.bn3_tensor.clone()}
        output = {'rgb_fine': self.bn3_tensor.clone()}
        # full image psnr on fine rgb
        psnr = PSNR(dict_to_obj({'key': 'rgb_fine'}))
        metric = psnr(data, output)
        self.assertEqual(metric.shape, ())

    def tests_maskpsnr(self):
        data = {'img': self.bn3_tensor.clone(), 'mask': self.bn_tensor.clone()}
        output = {'rgb': self.bn3_tensor.clone()}
        # full image masked area psnr
        psnr = MaskPSNR(dict_to_obj({}))
        metric = psnr(data, output)
        self.assertEqual(metric.shape, ())

    def tests_maskpsnrcoarse(self):
        data = {'img': self.bn3_tensor.clone(), 'mask': self.bn_tensor.clone()}
        output = {'rgb_coarse': self.bn3_tensor.clone()}
        # full image masked area psnr on coarse rgb
        psnr = MaskPSNR(dict_to_obj({'key': 'rgb_coarse'}))
        metric = psnr(data, output)
        self.assertEqual(metric.shape, ())

    def tests_maskpsnrfine(self):
        data = {'img': self.bn3_tensor.clone(), 'mask': self.bn_tensor.clone()}
        output = {'rgb_fine': self.bn3_tensor.clone()}
        # full image masked area psnr on fine rgb
        psnr = MaskPSNR(dict_to_obj({'key': 'rgb_fine'}))
        metric = psnr(data, output)
        self.assertEqual(metric.shape, ())

    def tests_ssim(self):
        data = {'img': self.bn3_tensor.clone(), 'H': self.b_tensor * self.H, 'W': self.b_tensor * self.W}
        output = {'rgb': self.bn3_tensor.clone()}
        # full image SSIM
        ssim = SSIM(dict_to_obj({}))
        metric = ssim(data, output)
        self.assertEqual(metric.shape, ())

    def tests_ssimcoarse(self):
        data = {'img': self.bn3_tensor.clone(), 'H': self.b_tensor * self.H, 'W': self.b_tensor * self.W}
        output = {'rgb_coarse': self.bn3_tensor.clone()}
        # full image SSIM on coarse rgb
        ssim = SSIM(dict_to_obj({'key': 'rgb_coarse'}))
        metric = ssim(data, output)
        self.assertEqual(metric.shape, ())

    def tests_ssimfine(self):
        data = {'img': self.bn3_tensor.clone(), 'H': self.b_tensor * self.H, 'W': self.b_tensor * self.W}
        output = {'rgb_fine': self.bn3_tensor.clone()}
        # full image SSIM on fine rgb
        ssim = SSIM(dict_to_obj({'key': 'rgb_fine'}))
        metric = ssim(data, output)
        self.assertEqual(metric.shape, ())

    def tests_maskssim(self):
        data = {
            'img': self.bn3_tensor.clone(),
            'H': self.b_tensor * self.H,
            'W': self.b_tensor * self.W,
            'mask': self.bn_tensor.clone()
        }
        output = {'rgb': self.bn3_tensor.clone()}
        # full image SSIM on masked area
        ssim = MaskSSIM(dict_to_obj({}))
        metric = ssim(data, output)
        self.assertEqual(metric.shape, ())

    def tests_maskssimcoarse(self):
        data = {
            'img': self.bn3_tensor.clone(),
            'H': self.b_tensor * self.H,
            'W': self.b_tensor * self.W,
            'mask': self.bn_tensor.clone()
        }
        output = {'rgb_coarse': self.bn3_tensor.clone()}
        # full image SSIM on masked area on coarse rgb
        ssim = MaskSSIM(dict_to_obj({'key': 'rgb_coarse'}))
        metric = ssim(data, output)
        self.assertEqual(metric.shape, ())

    def tests_maskssimfine(self):
        data = {
            'img': self.bn3_tensor.clone(),
            'H': self.b_tensor * self.H,
            'W': self.b_tensor * self.W,
            'mask': self.bn_tensor.clone()
        }
        output = {'rgb_fine': self.bn3_tensor.clone()}
        # full image SSIM on masked area on fine rgb
        ssim = MaskSSIM(dict_to_obj({'key': 'rgb_fine'}))
        metric = ssim(data, output)
        self.assertEqual(metric.shape, ())
