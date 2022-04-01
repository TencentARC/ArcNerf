#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np
import torch

from arcnerf.geometry.poses import look_at
from arcnerf.render.camera import PerspectiveCamera
from arcnerf.visual.plot_3d import draw_3d_components
from tests import setup_test_config

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results', 'camera'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    def setUp(self):
        self.cfgs = setup_test_config()
        self.H, self.W = 480, 640
        self.focal = 1000.0
        self.skewness = 10.0
        self.origin = (0, 0, 0)
        self.cam_loc = np.array([1, 1, 1])
        self.radius = np.linalg.norm(self.cam_loc - np.array(self.origin))
        self.intrinsic, self.c2w = self.setup_params()
        self.camera = self.setup_camera()

    def setup_params(self):
        # intrinsic
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = self.focal
        intrinsic[1, 1] = self.focal
        intrinsic[0, 1] = self.skewness
        intrinsic[0, 2] = self.W / 2.0
        intrinsic[1, 2] = self.H / 2.0
        # extrinsic
        c2w = look_at(self.cam_loc, np.array(self.origin))

        return intrinsic, c2w

    def setup_camera(self):
        return PerspectiveCamera(self.intrinsic, self.c2w, self.H, self.W)

    @staticmethod
    def create_pixel_grid(W, H):
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        pixels = torch.stack([i, j], dim=-1)  # (W, H, 2)

        return pixels

    def tests_dtype(self):
        self.camera = PerspectiveCamera(self.intrinsic, self.c2w, self.H, self.W, dtype=torch.float64)
        ray_bundle = self.camera.get_rays()
        self.assertEqual(ray_bundle[0].dtype, torch.float64)
        self.camera = PerspectiveCamera(self.intrinsic, self.c2w, self.H, self.W, dtype=torch.float32)
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
        for N_rays in [1, 10, 100, 1000, 10000]:
            ray_bundle = self.camera.get_rays(N_rays=N_rays)
            self.assertEqual(ray_bundle[0].shape, (N_rays, 3))
            self.assertEqual(ray_bundle[1].shape, (N_rays, 3))
            self.assertEqual(len(ray_bundle[2]), N_rays)
        for N_rays in [1, 10, 100, 1000, 10000]:
            index = np.random.choice(range(0, self.W * self.H), N_rays, replace=False)  # N_rays in (HW)
            ind_x, ind_y = index // self.W, index % self.H
            index = np.concatenate([ind_x[:, None], ind_y[:, None]], axis=-1)
            self.assertEqual(index.shape, (N_rays, 2))
            ray_bundle = self.camera.get_rays(index=index)
            self.assertEqual(ray_bundle[0].shape, (N_rays, 3))
            self.assertEqual(ray_bundle[1].shape, (N_rays, 3))
            self.assertEqual(len(ray_bundle[2]), N_rays)

    def test_n_rays_visual(self):
        ray_bundle = self.camera.get_rays(N_rays=10, to_np=True)
        z_factor = 3
        file_path = osp.join(RESULT_DIR, 'sample_10_rays.png')
        draw_3d_components(
            self.c2w[None, :],
            points=np.array(self.origin)[None, :],
            rays=(ray_bundle[0], ray_bundle[1] * z_factor),
            sphere_radius=self.radius,
            sphere_origin=self.origin,
            title='Cam with 10 rays(z factor {})'.format(z_factor),
            save_path=file_path
        )

    def test_index_rays_visual(self):
        index = np.array([[0, 0]])
        ray_bundle = self.camera.get_rays(index=index, to_np=True)
        file_path = osp.join(RESULT_DIR, 'sample_(0,0)_rays.png')
        draw_3d_components(
            self.c2w[None, :],
            points=np.array(self.origin)[None, :],
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
            points=np.array(self.origin)[None, :],
            rays=(ray_bundle[0], ray_bundle[1]),
            title='Cam ray at (W-1,H-1)',
            save_path=file_path
        )

    @staticmethod
    def get_max_abs_error(a, b):
        return float((a - b).abs().max())


if __name__ == '__main__':
    unittest.main()
