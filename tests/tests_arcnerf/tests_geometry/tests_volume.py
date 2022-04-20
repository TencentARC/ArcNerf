#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

from arcnerf.geometry.volume import Volume
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.torch_utils import torch_to_np

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results', 'volume'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.side = 1.5
        cls.n_grid = 4
        cls.xlen = 1.0
        cls.ylen = 2.0
        cls.zlen = 1.5
        cls.radius = 2.0
        cls.origin = (0.5, 0.5, 0)

    def tests_center_volume(self):
        volume = Volume(n_grid=self.n_grid, side=self.side)
        self.assertEqual(torch_to_np(volume.get_origin()).shape, (3, ))
        self.assertEqual(torch_to_np(volume.get_corner()).shape, (8, 3))
        self.assertEqual(torch_to_np(volume.get_grid_pts()).shape, ((self.n_grid + 1)**3, 3))

        volume_dict = {
            'grid_pts': torch_to_np(volume.get_corner()),  # (8, 3)
            'lines': volume.get_bound_lines(),  # (2*6, 3)
            'faces': volume.get_bound_faces()  # (6, 4, 3)
        }
        file_path = osp.join(RESULT_DIR, 'center_volume_bound.png')
        draw_3d_components(
            volume=volume_dict,
            sphere_radius=self.radius,
            title='center volume with bound lines/faces',
            save_path=file_path
        )

        volume_dict = {
            'grid_pts': torch_to_np(volume.get_grid_pts()),  # (n+1^3, 3)
            'volume_pts': torch_to_np(volume.get_volume_pts()),  # (n^3, 3)
            'lines': volume.get_dense_lines(),  # 3(n+1)^3 * (2, 3)
            'faces': volume.get_dense_faces()  # (3(n+1)n^2, 4, 3)
        }
        file_path = osp.join(RESULT_DIR, 'center_volume_dense.png')
        draw_3d_components(
            volume=volume_dict,
            sphere_radius=self.radius,
            title='center volume with dense lines/faces',
            save_path=file_path
        )

    def tests_custom_volume(self):
        volume = Volume(n_grid=self.n_grid)
        volume.set_params(self.origin, None, self.xlen, self.ylen, self.zlen)

        volume_dict = {
            'grid_pts': torch_to_np(volume.get_corner()),  # (8, 3)
            'lines': volume.get_bound_lines(),  # (2*6, 3)
            'faces': volume.get_bound_faces()  # (3(n+1)n^2, 4, 3)
        }
        file_path = osp.join(RESULT_DIR, 'custom_volume_bound.png')
        draw_3d_components(
            volume=volume_dict,
            sphere_radius=self.radius,
            title='custom volume with bound lines/faces',
            save_path=file_path
        )

        volume_dict = {
            'grid_pts': torch_to_np(volume.get_grid_pts()),  # (n+1^3, 3)
            'volume_pts': torch_to_np(volume.get_volume_pts()),  # (n^3, 3)
            'lines': volume.get_dense_lines(),  # 3(n+1)^3 * (2, 3)
            'faces': volume.get_dense_faces()  # (3(n+1)n^2, 4, 3)
        }
        file_path = osp.join(RESULT_DIR, 'custom_volume_dense.png')
        draw_3d_components(
            volume=volume_dict,
            sphere_radius=self.radius,
            title='custom volume with dense lines/faces',
            save_path=file_path
        )
