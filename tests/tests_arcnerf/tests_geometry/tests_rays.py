#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np

from arcnerf.geometry import np_wrapper
from arcnerf.geometry.poses import look_at
from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.render.camera import PerspectiveCamera, equal_sample
from arcnerf.visual.plot_3d import draw_3d_components
from common.visual import get_combine_colors
from tests import setup_test_config

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results', 'rays'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfgs = setup_test_config()
        cls.H, cls.W = 480, 640
        cls.focal = 1000.0
        cls.skewness = 10.0
        cls.origin = (0, 0, 0)
        cls.cam_loc = np.array([1, 1, -1])
        cls.radius = np.linalg.norm(cls.cam_loc - np.array(cls.origin))
        cls.intrinsic, cls.c2w = cls.setup_params()
        cls.camera = cls.setup_camera()

        # get rays from camera
        cls.n_rays_w, cls.n_rays_h = 5, 3
        cls.z_min, cls.z_max = 0.5, 2.0
        cls.n_rays = cls.n_rays_w * cls.n_rays_h
        cls.ray_bundle = cls.get_rays()

    @classmethod
    def setup_params(cls):
        # intrinsic
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = cls.focal
        intrinsic[1, 1] = cls.focal
        intrinsic[0, 1] = cls.skewness
        intrinsic[0, 2] = cls.W / 2.0
        intrinsic[1, 2] = cls.H / 2.0
        # extrinsic
        c2w = look_at(cls.cam_loc, np.array(cls.origin))

        return intrinsic, c2w

    @classmethod
    def setup_camera(cls):
        return PerspectiveCamera(cls.intrinsic, cls.c2w, cls.H, cls.W)

    @classmethod
    def get_rays(cls):
        index = equal_sample(cls.n_rays_w, cls.n_rays_h, cls.W, cls.H)
        ray_bundle = cls.camera.get_rays(index=index, to_np=True)

        return ray_bundle

    def tests_ray_points(self):
        # get points by different depths
        n_pts = 5
        zvals = np.linspace(self.z_min, self.z_max, n_pts + 2)[1:-1]  # (n_pts, )
        zvals = np.repeat(zvals[None, :], self.n_rays, axis=0)  # (n_rays, n_pts)
        points = np_wrapper(
            get_ray_points_by_zvals, self.ray_bundle[0], self.ray_bundle[1], zvals
        )  # (n_rays, n_pts, 3)
        points = points.reshape(-1, 3)
        self.assertEqual(points.shape[0], self.n_rays * n_pts)
        points_all = np.concatenate([np.array(self.origin)[None, :], points], axis=0)
        point_colors = get_combine_colors(['green', 'red'], [1, points.shape[0]])
        ray_colors = get_combine_colors(['sky_blue'], [self.n_rays])

        file_path = osp.join(RESULT_DIR, 'rays_sample_points.png')
        draw_3d_components(
            self.c2w[None, :],
            intrinsic=self.intrinsic,
            points=points_all,
            point_colors=point_colors,
            rays=(self.ray_bundle[0], self.z_max * self.ray_bundle[1]),
            ray_colors=ray_colors,
            ray_linewidth=0.5,
            title='Each ray sample {} points'.format(n_pts),
            save_path=file_path
        )

    @staticmethod
    def get_max_abs_error(a, b):
        return float((a - b).abs().max())


if __name__ == '__main__':
    unittest.main()
