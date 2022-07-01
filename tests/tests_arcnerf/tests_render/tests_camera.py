#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np
import torch

from arcnerf.geometry.poses import look_at
from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.render.camera import PerspectiveCamera
from arcnerf.render.ray_helper import equal_sample
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.torch_utils import np_wrapper
from common.visual import get_combine_colors
from tests import setup_test_config

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results', 'camera'))
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
        return PerspectiveCamera(cls.intrinsic, cls.c2w, cls.W, cls.H)

    @staticmethod
    def create_pixel_grid(W, H):
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        pixels = torch.stack([i, j], dim=-1)  # (W, H, 2)

        return pixels

    def tests_dtype(self):
        self.camera = PerspectiveCamera(self.intrinsic, self.c2w, self.W, self.H, dtype=torch.float64)
        ray_bundle = self.camera.get_rays()
        self.assertEqual(ray_bundle[0].dtype, torch.float64)
        self.camera = PerspectiveCamera(self.intrinsic, self.c2w, self.W, self.H, dtype=torch.float32)
        ray_bundle = self.camera.get_rays()
        self.assertEqual(ray_bundle[0].dtype, torch.float32)

    def tests_proj(self):
        """Ray with different depths should be projected to the same position"""
        ray_bundle = self.camera.get_rays()  # ray in world coord
        for i in range(1, 10):  # any depth
            reproj_pixel = self.camera.proj_world_to_pixel(ray_bundle[0] + i * ray_bundle[1]).reshape(self.W, self.H, 2)
            ori_pixel = self.create_pixel_grid(self.W, self.H)
            self.assertTrue(
                torch.allclose(ori_pixel, reproj_pixel, atol=1e-2),
                'max_error {:.5f}'.format(self.get_max_abs_error(ori_pixel, reproj_pixel))
            )

    def tests_rescale(self):
        """Rescale camera params and project to different img size, check whether same proj location"""
        ray_bundle = self.camera.get_rays()  # ray in world coord
        for scale in [0.25, 0.5, 2.0, 4.0]:
            self.camera.rescale(scale)
            self.assertEqual(self.camera.get_intrinsic(torch_tensor=False)[0, 0], self.focal * scale)
            scale_w, scale_h = int(self.W * scale), int(self.H * scale)
            self.assertEqual(self.camera.get_wh()[0], scale_w)
            self.assertEqual(self.camera.get_wh()[1], scale_h)
            reproj_pixel = self.camera.proj_world_to_pixel(ray_bundle[0] + ray_bundle[1])
            reproj_pixel = reproj_pixel.reshape(self.W, self.H, 2)  # (W, H, 2)
            ori_pixel = self.create_pixel_grid(scale_w, scale_h)  # (scale * W, scale * H, 2)
            if scale < 1.0:
                step = int(1 / scale)
                self.assertTrue(torch.allclose(reproj_pixel[::step, ::step, :], ori_pixel, atol=1e-2))
            elif scale > 1.0:
                step = int(scale)
                self.assertTrue(torch.allclose(ori_pixel[::step, ::step, :], reproj_pixel, atol=1e-2))
            self.camera = self.setup_camera()  # set back default camera

    def tests_get_rays(self):
        ray_bundle = self.camera.get_rays()
        self.assertEqual(ray_bundle[0].shape, (self.H * self.W, 3))
        self.assertEqual(ray_bundle[1].shape, (self.H * self.W, 3))
        self.assertIsNone(ray_bundle[2])
        self.assertEqual(ray_bundle[3].shape, (self.H * self.W, 1))
        for n_rays in [1, 10, 100, 1000, 10000]:
            ray_bundle = self.camera.get_rays(n_rays=n_rays)
            self.assertEqual(ray_bundle[0].shape, (n_rays, 3))
            self.assertEqual(ray_bundle[1].shape, (n_rays, 3))
            self.assertEqual(len(ray_bundle[2]), n_rays)
        for n_rays in [1, 10, 100, 1000, 10000]:
            index = np.random.choice(range(0, self.W * self.H - 1), n_rays, replace=False)  # N_rays in (HW)
            ind_x, ind_y = index // self.W, index % self.H
            index = np.concatenate([ind_x[:, None], ind_y[:, None]], axis=-1)
            self.assertEqual(index.shape, (n_rays, 2))
            ray_bundle = self.camera.get_rays(index=index)
            self.assertEqual(ray_bundle[0].shape, (n_rays, 3))
            self.assertEqual(ray_bundle[1].shape, (n_rays, 3))
            self.assertEqual(len(ray_bundle[2]), n_rays)

    def tests_n_rays_visual(self):
        ray_bundle = self.camera.get_rays(n_rays=10, to_np=True)
        up = self.c2w[:3, 1][None, :]

        z_factor = 3
        rays_origin = np.concatenate([self.c2w[:3, 3][None, :], ray_bundle[0]], axis=0)
        rays_dir = np.concatenate([up, ray_bundle[1] * z_factor], axis=0)
        ray_colors = get_combine_colors(['maroon', 'blue'], [1, ray_bundle[0].shape[0]])

        file_path = osp.join(RESULT_DIR, 'sample_10_rays.png')
        draw_3d_components(
            self.c2w[None, :],
            points=np.array([self.origin]),
            rays=(rays_origin, rays_dir),
            ray_colors=ray_colors,
            sphere_radius=self.radius,
            sphere_origin=self.origin,
            title='Cam with 10 rays(z factor {})'.format(z_factor),
            save_path=file_path
        )

    def tests_index_rays_visual(self):
        index = np.array([[0, 0]])
        ray_bundle = self.camera.get_rays(index=index, to_np=True)
        file_path = osp.join(RESULT_DIR, 'sample_(0,0)_rays.png')
        draw_3d_components(
            self.c2w[None, :],
            points=np.array([self.origin]),
            rays=(ray_bundle[0], ray_bundle[1]),
            title='Cam ray at (0,0)',
            save_path=file_path
        )

        index = np.array([[self.W - 1, self.H - 1]])
        ray_bundle = self.camera.get_rays(index=index, to_np=True)
        file_path = osp.join(RESULT_DIR, 'sample_(W,H)_rays.png')
        draw_3d_components(
            self.c2w[None, :],
            intrinsic=self.intrinsic,
            points=np.array([self.origin]),
            rays=(ray_bundle[0], ray_bundle[1]),
            title='Cam ray at (W-1,H-1)',
            save_path=file_path
        )

    def tests_ray_points(self):
        n_rays_w, n_rays_h = 8, 6
        n_rays = n_rays_w * n_rays_h
        index = equal_sample(n_rays_w, n_rays_h, self.W, self.H)
        ray_bundle = self.camera.get_rays(index=index, to_np=True)

        # get points by different depths
        n_pts = 5
        z_max = 3
        zvals = np.linspace(0, z_max, n_pts + 2)[1:-1]  # (n_pts, )
        zvals = np.repeat(zvals[None, :], n_rays, axis=0)  # (n_rays, n_pts)
        points = np_wrapper(get_ray_points_by_zvals, ray_bundle[0], ray_bundle[1], zvals)  # (n_rays, n_pts, 3)
        points = points.reshape(-1, 3)
        self.assertEqual(points.shape[0], n_rays * n_pts)
        points_all = np.concatenate([np.array([self.origin]), points], axis=0)
        point_colors = get_combine_colors(['green', 'red'], [1, points.shape[0]])
        ray_colors = get_combine_colors(['sky_blue'], [n_rays])

        file_path = osp.join(RESULT_DIR, 'rays_sample_points.png')
        draw_3d_components(
            self.c2w[None, :],
            intrinsic=self.intrinsic,
            points=points_all,
            point_colors=point_colors,
            rays=(ray_bundle[0], z_max * ray_bundle[1]),
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
