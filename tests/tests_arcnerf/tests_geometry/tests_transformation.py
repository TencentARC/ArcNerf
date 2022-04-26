#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
import torch

from arcnerf.geometry.poses import invert_poses
from arcnerf.geometry.projection import pixel_to_cam, cam_to_world, world_to_pixel
from arcnerf.geometry.transformation import normalize, rotate_points, get_rotate_matrix_from_vec, rotate_matrix
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

    def tests_get_rotate_matrix_from_vec(self):
        vec_1 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        vec_2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        mat = get_rotate_matrix_from_vec(vec_1, vec_2)
        p_1 = torch.tensor([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0]]).unsqueeze(1)
        p_2_gt = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, -2.0, 0.0]]).unsqueeze(1)
        p_2 = rotate_points(p_1, mat, rotate_only=True)
        self.assertTrue(torch.allclose(p_2, p_2_gt))

    def tests_rotation_reproj(self):
        xyz_cam = pixel_to_cam(self.pixels, self.depth, self.intrinsic)
        xyz_world = cam_to_world(xyz_cam, self.c2w)
        avg_pose = torch.rand(size=(xyz_world.shape[0], 3, 3))
        rot_mat = torch.repeat_interleave(torch.eye(4)[None], xyz_world.shape[0], 0)
        rot_mat[:, :3, :3] = torch.inverse(avg_pose)[:, :3, :3]  # (n, 4, 4)
        # rotate c2w and pts together, test reproj
        c2w_rotate = rotate_matrix(rot_mat, self.c2w)
        xyz_world_rotate = rotate_points(xyz_world, rot_mat)
        # reproj
        xyz_pixel_rotate = world_to_pixel(xyz_world_rotate, self.intrinsic, invert_poses(c2w_rotate))
        self.assertTrue(torch.allclose(self.pixels, xyz_pixel_rotate, atol=1e-2))

    @staticmethod
    def get_max_abs_error(a, b):
        return float((a - b).abs().max())


if __name__ == '__main__':
    unittest.main()
