#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.geometry.projection import (
    cam_to_pixel, cam_to_world, pixel_to_cam, pixel_to_world, world_to_cam, world_to_pixel
)
from tests.tests_arcnerf.tests_geometry import TestGeomDict


class TestDict(TestGeomDict):

    def setUp(self):
        super().setUp()

    def tests_points_transform(self):
        xyz_cam = pixel_to_cam(self.pixels, self.depth, self.intrinsic)
        self.assertEqual(xyz_cam.shape, (self.batch_size, self.H * self.W, 3))
        xyz_world = cam_to_world(xyz_cam, self.c2w)
        self.assertEqual(xyz_world.shape, (self.batch_size, self.H * self.W, 3))
        xyz_world_direct = pixel_to_world(self.pixels, self.depth, self.intrinsic, self.c2w)
        self.assertTrue(torch.allclose(xyz_world, xyz_world_direct))

        # Seems like w2c invert increase error at above 1e-3 level
        xyz_cam_from_world = world_to_cam(xyz_world, self.get_w2c())
        self.assertTrue(torch.allclose(xyz_cam, xyz_cam_from_world, atol=1e-5))
        # accumulated error at pixel level is above 1e-2
        pixel_from_cam = cam_to_pixel(xyz_cam, self.intrinsic)
        self.assertTrue(torch.allclose(self.pixels, pixel_from_cam, atol=1e-3))
        pixel_from_world = world_to_pixel(xyz_world, self.intrinsic, self.get_w2c())
        self.assertTrue(
            torch.allclose(self.pixels, pixel_from_world, atol=1e-2),
            'max_error {:.5f}'.format(self.get_max_abs_error(self.pixels, pixel_from_world))
        )

    @staticmethod
    def get_max_abs_error(a, b):
        return float((a - b).abs().max())


if __name__ == '__main__':
    unittest.main()
