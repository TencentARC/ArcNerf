#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np
import torch

from arcnerf.geometry.poses import invert_poses, look_at, get_sphere_line, get_spiral_line
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
        up = np.array([0, 1, 0])
        view_mat = look_at(cam_loc, point, up)
        rays_d = normalize(point - cam_loc)
        self.assertTrue(np.allclose(view_mat[:3, 3], cam_loc))

        file_path = osp.join(RESULT_DIR, 'look_at.png')
        draw_3d_components(
            view_mat[None],
            points=point[None, :],
            rays=(cam_loc[None, :], rays_d[None, :]),
            sphere_radius=4,
            title='look at camera from (1,2,3) to (0,0,0)',
            save_path=file_path
        )

    def tests_sphere_line(self):
        file_path = osp.join(RESULT_DIR, 'sphere_line.png')
        sphere_lines = []
        origin = (5, 5, 0)
        for v in [-0.5, 0, 0.5, 0.8]:
            sphere_lines.append(get_sphere_line(4, v=v, origin=origin))
        draw_3d_components(
            sphere_radius=4,
            sphere_origin=origin,
            lines=sphere_lines,
            title='sphere_line_ori(5,5,0)',
            save_path=file_path
        )

    def tests_spiral_line(self):
        file_path = osp.join(RESULT_DIR, 'spiral_lines.png')
        origin = (5, 5, 0)
        spiral_lines = [get_spiral_line(4, u_start=0.25, v_range=(0.75, -0.25), origin=origin)]
        draw_3d_components(
            sphere_radius=4,
            sphere_origin=origin,
            lines=spiral_lines,
            title='spiral_lines_ustart_0.25_vrange(0.75, -0.25)_origin(5,5,0)',
            save_path=file_path
        )

    @staticmethod
    def get_max_abs_error(a, b):
        return float((a - b).abs().max())


if __name__ == '__main__':
    unittest.main()
