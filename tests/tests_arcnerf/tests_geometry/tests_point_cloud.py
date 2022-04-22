#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import os
import os.path as osp
import unittest

import numpy as np

from arcnerf.geometry.point_cloud import save_point_cloud
from arcnerf.geometry.volume import Volume
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.torch_utils import torch_to_np
from common.visual import get_colors_from_cm

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results', 'point_cloud'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.side = 1.5
        cls.n_grid = [128, 256]
        cls.radius = 0.5
        cls.hourglass_h = 1.0
        cls.level = 0
        cls.max_pts = 200000

    def make_sphere_sdf(self, volume_pts):
        """make the sphere sdf with radius from volume_pts"""
        return self.radius - np.linalg.norm(volume_pts, axis=-1)

    def make_hourglass_sdf(self, volume_pts, n_grid):
        """make the hourglass sdf with radius from volume_pts"""
        sigma = abs(volume_pts[:, 1])**2 - (volume_pts[:, 0]**2 + volume_pts[:, 2]**2)
        sigma = sigma.reshape((n_grid, n_grid, n_grid))
        offset = int(n_grid * (1 - (self.hourglass_h / self.side)) / 2)
        sigma[:, :offset, :] = -1.0
        sigma[:, -offset:, :] = -1.0

        return sigma.reshape(-1)

    def run_point_cloud(self, type='sphere'):
        """Simulate a object with sigma >0 inside, <0 outside. Extract point"""
        assert type in ['sphere', 'hourglass'], 'Invalid object'
        object_dir = osp.join(RESULT_DIR, type)
        os.makedirs(object_dir, exist_ok=True)

        for n_grid in self.n_grid:
            grid_dir = osp.join(object_dir, 'ngrid{}'.format(n_grid))
            os.makedirs(grid_dir, exist_ok=True)

            volume = Volume(n_grid=n_grid, side=self.side)
            volume_pts = torch_to_np(volume.get_volume_pts())  # (n^3, 3)
            volume_dict = {
                'grid_pts': torch_to_np(volume.get_corner()),
                'lines': volume.get_bound_lines(),
                'faces': volume.get_bound_faces()
            }

            # simulate a object sdf
            if type == 'sphere':
                sigma = self.make_sphere_sdf(volume_pts)  # (n^3, )
            else:
                sigma = self.make_hourglass_sdf(volume_pts, n_grid)  # (n^3, )
            n_pts = sigma.shape[0]
            rgb = get_colors_from_cm(1024, to_np=True)
            rgb = rgb.repeat(max(1, math.ceil(float(n_pts) / 1024.0)), 0)[:n_pts]  # (n^3, 3)

            valid_sigma = (sigma >= self.level)  # inside pts (n^3,)
            valid_pts = volume_pts[valid_sigma]  # (n_valid, 3)
            valid_rgb = rgb[valid_sigma]

            # export point cloud
            pc_file = osp.join(grid_dir, 'pc_ngrid{}.ply'.format(n_grid))
            save_point_cloud(pc_file, valid_pts, valid_rgb)

            # simplify pts
            n_pts = valid_pts.shape[0]
            if n_pts > self.max_pts:  # in case to many point to draw
                choice = np.random.choice(range(n_pts), self.max_pts, replace=False)
                valid_pts = valid_pts[choice]
                valid_rgb = valid_rgb[choice]

            # draw pts in plotly
            file_path = osp.join(grid_dir, 'pc_ngrid{}.png'.format(n_grid))
            draw_3d_components(
                points=valid_pts,
                point_colors=valid_rgb,
                volume=volume_dict,
                title='valid {} pts from volume'.format(valid_pts.shape[0]),
                save_path=file_path,
                plotly=True,
                plotly_html=True
            )

    def tests_point_cloud_sphere(self):
        self.run_point_cloud('sphere')

    def tests_point_cloud_hourglass(self):
        self.run_point_cloud('hourglass')
