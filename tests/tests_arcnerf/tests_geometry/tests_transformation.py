#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
import torch

from arcnerf.geometry.poses import invert_poses
from arcnerf.geometry.projection import pixel_to_cam
from arcnerf.geometry.transformation import normalize, rotate_points
from tests.tests_arcnerf.tests_geometry import TestGeomDict


class TestDict(TestGeomDict):

    @classmethod
    def setUpClass(cls):
        super(TestDict, cls).setUpClass()

    def tests_normalization(self):
        vec = pixel_to_cam(self.pixels, self.depth, self.intrinsic)
        norm_vec = normalize(vec)
        self.assertTrue(torch.allclose(torch.ones(size=norm_vec.shape[:2]), torch.norm(norm_vec, dim=-1)))
        vec_np = vec.numpy()
        norm_vec_np = normalize(vec_np)
        self.assertTrue(np.allclose(np.ones(shape=norm_vec_np.shape[:2]), np.linalg.norm(norm_vec_np, axis=-1)))

    def tests_rotate_points(self):
        points = torch.rand(size=(self.batch_size, 1000, 3))
        rot = torch.eye(4)
        rot[:3, :4] = torch.rand(size=(3, 4))
        rot = torch.repeat_interleave(rot.unsqueeze(0), self.batch_size, dim=0)
        inv_rot = invert_poses(rot)
        points_test = rotate_points(rotate_points(points, rot), inv_rot)
        self.assertTrue(torch.allclose(points, points_test, atol=1e-3))

    @staticmethod
    def get_max_abs_error(a, b):
        return float((a - b).abs().max())


if __name__ == '__main__':
    unittest.main()
