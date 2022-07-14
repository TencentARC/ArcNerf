#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np
import torch

from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from arcnerf.geometry.volume import Volume
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.torch_utils import torch_to_np, np_wrapper
from common.visual import get_combine_colors, get_colors

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

    def tests_ray_volume(self):
        """Test some ray volume interaction function"""
        volume = Volume(n_grid=self.n_grid, side=self.side)
        rays_o = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        rays_d = np.array([[-1.5, -1.0, -0.8]], dtype=np.float32)
        rays_d = normalize(rays_d)

        n_pts = 15
        zvals = np.linspace(0.0, 4.0, n_pts, dtype=np.float32)[None]  # (1, n_pts)
        pts = np_wrapper(get_ray_points_by_zvals, rays_o, rays_d, zvals).reshape(-1, 3)  # (n_pts, 3)
        pts_in_boundary = np_wrapper(volume.check_pts_in_grid_boundary, pts, volume.get_corner())  # (n_in, )

        # assign color to pts
        pts_colors = get_combine_colors(['red'], [pts.shape[0]])
        green_color = get_colors('green', to_int=False, to_np=True)
        pts_colors[pts_in_boundary] = green_color

        # get voxel idx
        voxel_idx, valid_idx = np_wrapper(volume.get_voxel_idx_from_xyz, pts)
        self.assertTrue(np.all(np.equal(valid_idx, pts_in_boundary)))

        # get valid grid_pts
        grid_pts_valid = np_wrapper(volume.get_grid_pts_by_voxel_idx, voxel_idx[valid_idx])  # (n_in, 8, 3)
        grid_pts_weights_valid = np_wrapper(volume.cal_weights_to_grid_pts, pts[valid_idx], grid_pts_valid)  # (n_in, 8)
        self.assertTrue(np.all(grid_pts_weights_valid > 0))

        # check by direct method
        voxel_grid = np_wrapper(volume.get_voxel_grid_info_from_xyz, pts)
        self.assertTrue(np.allclose(voxel_grid[-1], grid_pts_weights_valid))

        # get lines and draw lines with weights as width
        lines = []
        line_widths = []
        line_colors = []
        for idx in range(grid_pts_valid.shape[0]):
            for grid_pts_idx in range(grid_pts_valid.shape[1]):
                line = np.concatenate([pts[valid_idx][idx][None], grid_pts_valid[idx][grid_pts_idx][None]])
                lines.append(line)
                weight = grid_pts_weights_valid[idx][grid_pts_idx]
                line_widths.append(weight * 10.0)
                line_color = get_colors('red', to_int=False, to_np=True)[None]
                line_colors.append(line_color)
        line_colors = np.concatenate(line_colors, axis=0)

        # draw
        volume_dict = {
            'grid_pts': torch_to_np(volume.get_grid_pts()),  # (n+1^3, 3)
            'lines': volume.get_dense_lines(),  # 3(n+1)^3 * (2, 3)
        }

        file_path = osp.join(RESULT_DIR, 'volume_pts_interpolation.png')
        draw_3d_components(
            points=pts,
            point_colors=pts_colors,
            lines=lines,
            line_widths=line_widths,
            line_colors=line_colors,
            volume=volume_dict,
            title='volume with ray pts interpolation',
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

    def tests_get_collect_grid_pts_by_voxel_idx(self):
        volume = Volume(n_grid=512, side=self.side)
        batch_size = 4096
        pts = torch.rand((batch_size, 3))

        # get voxel idx
        voxel_idx, valid_idx = np_wrapper(volume.get_voxel_idx_from_xyz, pts)

        # collect by voxel_idx
        grid_pts_1 = np_wrapper(volume.collect_grid_pts_by_voxel_idx, voxel_idx[valid_idx])

        # get by voxel_idx
        grid_pts_2 = np_wrapper(volume.get_grid_pts_by_voxel_idx, voxel_idx[valid_idx])

        self.assertTrue(np.allclose(grid_pts_1, grid_pts_2))
