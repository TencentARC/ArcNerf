#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np

from arcnerf.datasets import get_dataset
from arcnerf.datasets.transform.augmentation import get_transforms
from arcnerf.geometry.poses import average_poses, generate_cam_pose_on_sphere
from arcnerf.geometry.sphere import get_uv_from_pos
from arcnerf.geometry.transformation import normalize
from arcnerf.visual.plot_3d import draw_3d_components
from common.visual import get_colors
from tests import setup_test_config

MODE = 'train'
RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    def setUp(self):
        self.cfgs = setup_test_config()
        self.dataset_type = getattr(self.cfgs.dataset, MODE).type
        self.dataset = self.setup_dataset()
        self.c2w = self.get_cameras()  # (n, 4, 4)
        self.n_cam = self.c2w.shape[0]

        self.spec_result_dir = osp.abspath(osp.join(RESULT_DIR, self.dataset_type))
        os.makedirs(self.spec_result_dir, exist_ok=True)

    def setup_dataset(self):
        transforms, _ = get_transforms(getattr(self.cfgs.dataset, MODE))
        dataset = get_dataset(self.cfgs.dataset, self.cfgs.dir.data_dir, None, MODE, transforms)

        return dataset

    def get_cameras(self):
        c2w = []
        for sample in self.dataset:
            c2w.append(sample['c2w'][None, ...])
        c2w = np.concatenate(c2w, axis=0)  # (n, 4, 4)

        return c2w

    def tests_get_dataset(self):
        self.assertIsInstance(self.dataset[0], dict)

    def tests_vis_cameras(self):
        origin = (0, 0, 0)
        # combine avg pose with different color
        avg_pose = average_poses(self.c2w)[None, :]  # (1, 4, 4)
        c2w = np.concatenate([self.c2w, avg_pose], axis=0)  # (n+1, 4, 4)
        cam_colors = np.concatenate([
            np.repeat(get_colors(color='red', to_int=False, to_np=True)[None, :], self.n_cam, axis=0),
            get_colors('maroon', to_int=False, to_np=True)[None, :]
        ])  # (n+1, 3)
        cam_loc = np.concatenate([self.c2w[:, :3, 3], avg_pose[:, :3, 3]])  # (n+1, 3)
        rays_d = normalize(np.array(origin)[None, :] - cam_loc)  # (n+1, 3)
        rays_colors = np.concatenate([
            np.repeat(get_colors(color='blue', to_int=False, to_np=True)[None, :], self.n_cam, axis=0),
            get_colors('navy', to_int=False, to_np=True)[None, :]
        ])  # (n+1, 3)
        # mean sphere radius
        radius = np.linalg.norm(avg_pose[0, :3, 3])

        file_path = osp.join(self.spec_result_dir, '{}_vis_camera.png'.format(self.dataset_type))
        draw_3d_components(
            c2w,
            cam_colors=cam_colors,
            points=np.array(origin)[None, :],
            rays=(cam_loc, rays_d),
            ray_colors=rays_colors,
            sphere_radius=radius,
            sphere_origin=origin,
            title='{} Cam position'.format(self.dataset_type),
            save_path=file_path,
        )

    def tests_create_test_cam_path(self):
        n_test_cam = 25
        origin = (0, 0, 0)
        avg_pose = average_poses(self.c2w)[None, :]  # (1, 4, 4)

        u_start, v_ratio, radius = get_uv_from_pos(avg_pose[0, :3, 3], origin)
        _, v_min, _ = get_uv_from_pos(self.c2w[np.argmin(self.c2w[:, 1, 3]), :3, 3], origin)
        _, v_max, _ = get_uv_from_pos(self.c2w[np.argmax(self.c2w[:, 1, 3]), :3, 3], origin)

        modes = ['circle', 'spiral']
        for mode in modes:
            file_path = osp.join(self.spec_result_dir, 'test_cam_mode_{}.png'.format(mode))
            c2w_test = generate_cam_pose_on_sphere(
                mode,
                radius,
                n_test_cam,
                u_start=u_start,
                v_ratio=v_ratio,
                v_range=(v_max, v_min),
                origin=origin,
                n_rot=3,
                close=False  # just for test, should be true for actual visual
            )
            c2w = np.concatenate([c2w_test, avg_pose], axis=0)  # (n+1, 4, 4)
            cam_colors = np.concatenate([
                np.repeat(get_colors(color='red', to_int=False, to_np=True)[None, :], n_test_cam, axis=0),
                get_colors('maroon', to_int=False, to_np=True)[None, :]
            ])  # (n+1, 3)
            cam_loc = np.concatenate([c2w_test[:, :3, 3], avg_pose[:, :3, 3]])  # (n+1, 3)
            rays_d = normalize(np.array(origin)[None, :] - cam_loc)  # (n+1, 3)
            rays_colors = np.concatenate([
                np.repeat(get_colors(color='blue', to_int=False, to_np=True)[None, :], n_test_cam, axis=0),
                get_colors('navy', to_int=False, to_np=True)[None, :]
            ])  # (n+1, 3)
            draw_3d_components(
                c2w,
                cam_colors=cam_colors,
                points=np.array(origin)[None, :],
                rays=(cam_loc, rays_d),
                ray_colors=rays_colors,
                sphere_radius=radius,
                sphere_origin=origin,
                lines=[cam_loc[:n_test_cam]],
                title='Cam pos on sphere. Mode: {}'.format(mode),
                save_path=file_path
            )


if __name__ == '__main__':
    unittest.main()
