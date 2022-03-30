#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np
import torch

from arcnerf.geometry.poses import invert_poses, look_at
from arcnerf.geometry.transformation import normalize
from arcnerf.visual.plot_3d import draw_3d_components
from tests.tests_arcnerf.tests_geometry import TestGeomDict

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(TestGeomDict):

    def setUp(self):
        super().setUp()

    def tests_invert_pose(self):
        c2w_new = invert_poses(invert_poses(self.c2w))
        self.assertTrue(torch.allclose(self.c2w, c2w_new, atol=1e-3))
        c2w_np = self.c2w.numpy()
        c2w_np_new = invert_poses(invert_poses(c2w_np))
        self.assertTrue(np.allclose(c2w_np, c2w_np_new, atol=1e-3))

    def tests_look_at(self):
        cam_loc = np.array([1.0, 2.0, 3.0])
        point = np.array([0.0, 0.0, 0.0])
        up = np.array([0.2, 0.3, 0.4])
        view_mat = look_at(cam_loc, point, up)
        rays_d = normalize(point - cam_loc)
        self.assertTrue(np.allclose(view_mat[:3, 3], cam_loc))

        file_path = osp.join(RESULT_DIR, 'look_at.png')
        draw_3d_components(
            view_mat[None],
            points=point[None, :],
            rays=(cam_loc[None, :], rays_d[None, :]),
            title='look at camera from (1,2,3) to (0,0,0)',
            save_path=file_path
        )

    @staticmethod
    def get_max_abs_error(a, b):
        return float((a - b).abs().max())


if __name__ == '__main__':
    unittest.main()
