#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
import torch

from arcnerf.geometry.transformation import (
    cam_to_pixel, cam_to_world, invert_pose, normalize, pixel_to_cam, pixel_to_world, world_to_cam, world_to_pixel
)
from tests import setup_test_config


class TestDict(unittest.TestCase):

    def setUp(self):
        self.cfgs = setup_test_config()
        self.H, self.W = 480, 640
        self.batch_size = 2
        self.intrinsic, self.c2w = self.setup_params()
        self.pixels, self.depth = self.set_pixels()

    def setup_params(self):
        intrinsic = torch.eye(3, dtype=torch.float32)
        intrinsic[0, 2] = self.W / 2.0
        intrinsic[1, 2] = self.H / 2.0
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, :3] = torch.rand(size=(3, 3))
        intrinsic = torch.repeat_interleave(intrinsic[None, ...], self.batch_size, dim=0)
        c2w = torch.repeat_interleave(c2w[None, ...], self.batch_size, dim=0)

        return intrinsic, c2w

    def set_pixels(self):
        i, j = torch.meshgrid(
            torch.linspace(0, self.W - 1, self.W), torch.linspace(0, self.H - 1, self.H)
        )  # i, j: (W, H)
        pixels = torch.repeat_interleave(
            torch.stack([i, j]).reshape(-1, 2).unsqueeze(0), self.batch_size, dim=0
        )  # (2, WH, 2)
        depth = torch.ones(size=(self.batch_size, pixels.shape[1]))

        return pixels, depth

    def get_w2c(self):
        w2c = invert_pose(self.c2w)

        return w2c

    def tests_points_transform(self):
        xyz_cam = pixel_to_cam(self.pixels, self.depth, self.intrinsic)
        self.assertEqual(xyz_cam.shape, (self.batch_size, self.H * self.W, 3))
        xyz_world = cam_to_world(xyz_cam, self.c2w)
        self.assertEqual(xyz_world.shape, (self.batch_size, self.H * self.W, 3))
        xyz_world_direct = pixel_to_world(self.pixels, self.depth, self.intrinsic, self.c2w)
        self.assertTrue(torch.allclose(xyz_world, xyz_world_direct))

        # Seems like w2c invert increase error at max 1e-3 level
        xyz_cam_from_world = world_to_cam(xyz_world, self.get_w2c())
        self.assertTrue(torch.allclose(xyz_cam, xyz_cam_from_world, atol=1e-3))
        pixel_from_cam = cam_to_pixel(xyz_cam, self.intrinsic)
        self.assertTrue(torch.allclose(self.pixels, pixel_from_cam, atol=1e-3))
        pixel_from_world = world_to_pixel(xyz_world, self.intrinsic, self.get_w2c())
        self.assertTrue(
            torch.allclose(self.pixels, pixel_from_world, atol=1e-1),
            'max_error {:.2f}'.format(self.get_max_abs_error(self.pixels, pixel_from_world))
        )

    def tests_invert_pose(self):
        c2w_new = invert_pose(invert_pose(self.c2w))
        self.assertTrue(torch.allclose(self.c2w, c2w_new, atol=1e-3))
        c2w_np = self.c2w.numpy()
        c2w_np_new = invert_pose(invert_pose(c2w_np))
        self.assertTrue(np.allclose(c2w_np, c2w_np_new, atol=1e-3))

    def tests_normalization(self):
        vec = pixel_to_cam(self.pixels, self.depth, self.intrinsic)
        norm_vec = normalize(vec)
        self.assertTrue(torch.allclose(torch.ones(size=norm_vec.shape[:2]), torch.norm(norm_vec, dim=-1)))
        vec_np = vec.numpy()
        norm_vec_np = normalize(vec_np)
        self.assertTrue(np.allclose(np.ones(shape=norm_vec_np.shape[:2]), np.linalg.norm(norm_vec_np, axis=-1)))

    @staticmethod
    def get_max_abs_error(a, b):
        return float((a - b).abs().max())


if __name__ == '__main__':
    unittest.main()
