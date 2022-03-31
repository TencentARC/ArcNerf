#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np

from arcnerf.geometry.sphere import get_regular_sphere_line, get_sphere_line, get_spiral_line, get_uv_from_pos
from arcnerf.visual.plot_3d import draw_3d_components
from tests.tests_arcnerf.tests_geometry import TestGeomDict

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results', 'sphere'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(TestGeomDict):

    def setUp(self):
        super().setUp()
        self.radius = 4

    def tests_sphere_line(self):
        file_path = osp.join(RESULT_DIR, 'sphere_line.png')
        sphere_lines = []
        origin = (5, 5, 0)
        for v in [-0.5, 0, 0.5, 0.8]:
            sphere_lines.append(get_sphere_line(self.radius, v_ratio=v, origin=origin))
        draw_3d_components(
            sphere_radius=self.radius,
            sphere_origin=origin,
            lines=sphere_lines,
            title='sphere_line_ori(5,5,0)',
            save_path=file_path
        )

    def tests_regular_sphere_line(self):
        file_path = osp.join(RESULT_DIR, 'regular_sphere_line.png')
        origin = (5, 5, 0)
        regular_sphere_lines = get_regular_sphere_line(self.radius, n_rot=5, origin=origin, concat=False)
        draw_3d_components(
            sphere_radius=self.radius,
            sphere_origin=origin,
            lines=regular_sphere_lines,
            title='regular_sphere_line_ori(5,5,0)',
            save_path=file_path
        )

    def tests_spiral_line(self):
        file_path = osp.join(RESULT_DIR, 'spiral_lines.png')
        origin = (5, 5, 0)
        spiral_lines = [get_spiral_line(self.radius, u_start=0.25, v_range=(0.75, -0.25), origin=origin)]
        draw_3d_components(
            sphere_radius=self.radius,
            sphere_origin=origin,
            lines=spiral_lines,
            title='spiral_lines_ustart_0.25_vrange(0.75, -0.25)_origin(5,5,0)',
            save_path=file_path
        )

    def tests_get_uv_from_pos(self):
        file_path = osp.join(RESULT_DIR, 'get_uv_line.png')
        origin = (5, 5, 0)
        pos = np.array([6, 7, 2])
        u_start, v_ratio, radius = get_uv_from_pos(pos, origin)
        line = [get_sphere_line(radius, u_start, v_ratio, origin, n_pts=10, close=False)]
        draw_3d_components(
            points=np.concatenate([np.array(origin)[None, :], pos[None, :]]),
            sphere_radius=radius,
            sphere_origin=origin,
            lines=line,
            title='get_uv_from_pos(6,7,2)_from_origin(5,5,0)',
            save_path=file_path
        )

    @staticmethod
    def get_max_abs_error(a, b):
        return float((a - b).abs().max())


if __name__ == '__main__':
    unittest.main()
