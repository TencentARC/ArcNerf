#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
import torch

from arcnerf.render.camera import PerspectiveCamera
from tests import setup_test_config


class TestDict(unittest.TestCase):

    def setUp(self):
        self.cfgs = setup_test_config()
        self.H, self.W = 480, 640
        self.focal = 1000.0
        self.skewness = 10.0
        self.intrinsic, self.c2w = self.setup_params()
        self.camera = self.setup_default_camera()

    def setup_params(self):
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = self.focal
        intrinsic[1, 1] = self.focal
        intrinsic[0, 1] = self.skewness
        intrinsic[0, 2] = self.W / 2.0
        intrinsic[1, 2] = self.H / 2.0
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :4] = np.random.rand(3, 4)

        return intrinsic, c2w

    def setup_default_camera(self):
        return PerspectiveCamera(self.intrinsic, self.c2w, self.H, self.W)

    @staticmethod
    def create_pixel_grid(W, H):
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        pixels = torch.stack([i, j], dim=-1)  # (W, H, 2)

        return pixels

    def tests_get_rays(self):
        ray_bundle = self.camera.get_rays()
        self.assertEqual(ray_bundle[0].shape, ray_bundle[1].shape)
        self.assertEqual(ray_bundle[0].shape[0], self.H * self.W)

    def tests_dtype(self):
        self.camera = PerspectiveCamera(self.intrinsic, self.c2w, self.H, self.W, dtype=torch.float64)
        ray_bundle = self.camera.get_rays()
        self.assertEqual(ray_bundle[0].dtype, torch.float64)
        self.camera = PerspectiveCamera(self.intrinsic, self.c2w, self.H, self.W, dtype=torch.float32)
        ray_bundle = self.camera.get_rays()
        self.assertEqual(ray_bundle[0].dtype, torch.float32)

    def tests_proj(self):
        ray_bundle = self.camera.get_rays()  # ray in world coord
        for i in range(1, 10):  # any depth
            reproj_pixel = self.camera.proj_world_to_pixel(ray_bundle[0] + i * ray_bundle[1]).reshape(self.W, self.H, 2)
            ori_pixel = self.create_pixel_grid(self.W, self.H)
            self.assertTrue(
                torch.allclose(ori_pixel, reproj_pixel, atol=1e-2),
                'max_error {:.5f}'.format(self.get_max_abs_error(ori_pixel, reproj_pixel))
            )

    def tests_rescale(self):
        """Rescale camera params and proj, check whether same proj location"""
        ray_bundle = self.camera.get_rays()  # ray in world coord
        for scale in [0.25, 0.5, 2.0, 4.0]:
            self.camera.rescale(scale)
            self.assertEqual(self.camera.get_intrinsic(torch_tensor=False)[0, 0], self.focal * scale)
            scale_w, scale_h = int(self.W * scale), int(self.H * scale)
            self.assertEqual(self.camera.get_wh()[0], scale_w)
            self.assertEqual(self.camera.get_wh()[1], scale_h)
            reproj_pixel = self.camera.proj_world_to_pixel(ray_bundle[0] + ray_bundle[1])
            reproj_pixel = reproj_pixel.reshape(self.W, self.H, 2)  # (W, H, 2)
            ori_pixel = self.create_pixel_grid(scale_w, scale_h)  # (scale * W, scale * H, 2)
            if scale < 1.0:
                step = int(1 / scale)
                self.assertTrue(torch.allclose(reproj_pixel[::step, ::step, :], ori_pixel, atol=1e-2))
            elif scale > 1.0:
                step = int(scale)
                self.assertTrue(torch.allclose(ori_pixel[::step, ::step, :], reproj_pixel, atol=1e-2))
            self.camera = self.setup_default_camera()  # set back default camera

    @staticmethod
    def get_max_abs_error(a, b):
        return float((a - b).abs().max())


if __name__ == '__main__':
    unittest.main()
