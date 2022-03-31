#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np
import torch

from arcnerf.geometry.poses import (
    average_poses,
    center_poses,
    generate_cam_pose_on_sphere,
    look_at,
    invert_poses,
)
from arcnerf.geometry.transformation import normalize
from arcnerf.visual.plot_3d import draw_3d_components
from common.visual import get_colors
from tests.tests_arcnerf.tests_geometry import TestGeomDict

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(TestGeomDict):

    def setUp(self):
        super().setUp()
        self.radius = 4
        self.n_cam = 25

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
            sphere_radius=self.radius,
            title='look at camera from (1,2,3) to (0,0,0)',
            save_path=file_path
        )

    def tests_generate_cam_pose_on_sphere(self):
        u_start = 0  # (0, 1)
        v_ratio = 0.5  # (-1, 1)
        # custom case
        # look_at_point = np.array([1.0, 1.0, 0.0])  # (3, )
        # origin = (5, 5, 0)
        # these are centered on origin
        look_at_point = np.array([0.0, 0.0, 0.0])  # (3, )
        origin = (0, 0, 0)

        modes = ['random', 'regular', 'circle', 'spiral']
        for mode in modes:
            file_path = osp.join(RESULT_DIR, 'cam_path_mode_{}.png'.format(mode))
            c2w = generate_cam_pose_on_sphere(
                mode,
                self.radius,
                self.n_cam,
                u_start=u_start,
                v_ratio=v_ratio,
                origin=origin,
                look_at_point=look_at_point
            )
            cam_loc = c2w[:, :3, 3]  # (n, 3)
            rays_d = normalize(look_at_point[None, :] - cam_loc)  # (n, 3)
            track = [cam_loc] if mode != 'random' else []
            draw_3d_components(
                c2w,
                points=look_at_point[None, :],
                rays=(cam_loc, rays_d),
                sphere_radius=self.radius,
                sphere_origin=origin,
                lines=track,
                title='Cam pos on sphere. Mode: {}'.format(mode),
                save_path=file_path
            )

    def tests_regular_center_poses(self):
        n_cam = 12
        n_rot = 3
        look_at_point = np.array([5.0, 5.0, 0.0])  # (3, )
        origin = (0, 0, 0)
        c2w = generate_cam_pose_on_sphere(
            'regular',
            self.radius,
            n_cam,
            n_rot=n_rot,
            upper=True,
            close=False,
            origin=look_at_point,
            look_at_point=look_at_point
        )
        cam_colors = np.repeat(get_colors(color='red', to_int=False, to_np=True)[None, :], n_cam, axis=0)  # (n, 3)
        cam_loc = c2w[:, :3, 3]  # (n, 3)
        rays_d = normalize(look_at_point[None, :] - cam_loc)  # (n, 3)
        rays_colors = np.repeat(get_colors(color='blue', to_int=False, to_np=True)[None, :], n_cam, axis=0)  # (n, 3)

        # append avg poses
        avg_pose = average_poses(c2w)[None, :]  # (1, 4, 4)

        # center poses by real avg_pose, new centered poses is not one sphere center at (0,0,0)
        center_pose = center_poses(c2w)  # (n, 4, 4)
        cam_colors_all = np.concatenate([
            cam_colors,
            get_colors('maroon', to_int=False, to_np=True)[None, :],
            np.repeat(get_colors('black', to_int=False, to_np=True)[None, :], n_cam, axis=0)
        ])  # (n+1+n, 3)
        c2w_all = np.concatenate([c2w, avg_pose, center_pose], axis=0)  # (n+1+n, 4, 4)
        cam_loc_all = np.concatenate([cam_loc, avg_pose[:, :3, 3], center_pose[:, :3, 3]], axis=0)  # (n+1+n, 3)
        rays_d_all = np.concatenate([
            rays_d,
            normalize(look_at_point[None, :] - avg_pose[:, :3, 3]),
            normalize(np.array(list(origin))[None, :] - center_pose[:, :3, 3])
        ])  # (n+1+n, 3)
        rays_colors_all = np.concatenate([
            rays_colors,
            get_colors('navy', to_int=False, to_np=True)[None, :],
            np.repeat(get_colors('yellow', to_int=False, to_np=True)[None, :], n_cam, axis=0)
        ])  # (n+1+n, 3)
        points_all = np.concatenate([look_at_point[None, :], np.array(list(origin))[None, :]], axis=0)

        file_path = osp.join(RESULT_DIR, 'recenter_regular_poses.png')
        draw_3d_components(
            c2w_all,
            cam_colors=cam_colors_all,
            points=points_all,
            rays=(cam_loc_all, rays_d_all),
            ray_colors=rays_colors_all,
            sphere_radius=self.radius,
            sphere_origin=origin,
            title='Regular {} poses recentered'.format(n_cam),
            save_path=file_path
        )

    def tests_avg_and_center_poses(self):
        n_cam = 5
        look_at_point = np.array([5.0, 5.0, 0.0])  # (3, )
        origin = (0, 0, 0)
        # random pose on a sphere far from origin
        c2w = generate_cam_pose_on_sphere(
            'random', self.radius, n_cam, origin=look_at_point, look_at_point=look_at_point
        )  # (n, 4, 4)
        cam_colors = np.repeat(get_colors(color='red', to_int=False, to_np=True)[None, :], n_cam, axis=0)  # (n, 3)
        cam_loc = c2w[:, :3, 3]  # (n, 3)
        rays_d = normalize(look_at_point[None, :] - cam_loc)  # (n, 3)
        rays_colors = np.repeat(get_colors(color='blue', to_int=False, to_np=True)[None, :], n_cam, axis=0)  # (n, 3)

        # append avg poses
        avg_pose = average_poses(c2w)[None, :]  # (1, 4, 4)
        avg_pose_center = np.eye(4)[None, :]
        avg_pose_center[:, :3, 3] = look_at_point

        # combined color
        cam_colors_all = np.concatenate([
            cam_colors,
            get_colors('maroon', to_int=False, to_np=True)[None, :],
            np.repeat(get_colors('black', to_int=False, to_np=True)[None, :], n_cam, axis=0)
        ])  # (n+1+n, 3)
        rays_colors_all = np.concatenate([
            rays_colors,
            get_colors('navy', to_int=False, to_np=True)[None, :],
            np.repeat(get_colors('yellow', to_int=False, to_np=True)[None, :], n_cam, axis=0)
        ])  # (n+1+n, 3)

        # center poses, new center is the look at point, all center_pose is on sphere centered at 0
        center_pose = center_poses(c2w, look_at_point)  # (n, 4, 4)
        c2w_all = np.concatenate([c2w, avg_pose, center_pose], axis=0)  # (n+1+n, 4, 4)
        cam_loc_all = np.concatenate([cam_loc, avg_pose[:, :3, 3], center_pose[:, :3, 3]], axis=0)  # (n+1+n, 3)
        rays_d_all = np.concatenate([
            rays_d,
            normalize(look_at_point[None, :] - avg_pose[:, :3, 3]),
            normalize(np.array(list(origin))[None, :] - center_pose[:, :3, 3])
        ])  # (n+1+n, 3)
        points_all = np.concatenate([look_at_point[None, :], np.array(list(origin))[None, :]], axis=0)

        file_path = osp.join(RESULT_DIR, 'recenter_at_look_at.png')
        draw_3d_components(
            c2w_all,
            cam_colors=cam_colors_all,
            points=points_all,
            rays=(cam_loc_all, rays_d_all),
            ray_colors=rays_colors_all,
            sphere_radius=self.radius,
            sphere_origin=origin,
            title='Random {} cam on sphere, get avg pose and recenter by look at point'.format(n_cam),
            save_path=file_path
        )

        # center poses by real avg_pose, new centered poses is not one sphere center at (0,0,0)
        center_pose = center_poses(c2w)  # (n, 4, 4)
        c2w_all = np.concatenate([c2w, avg_pose, center_pose], axis=0)  # (n+1+n, 4, 4)
        cam_loc_all = np.concatenate([cam_loc, avg_pose[:, :3, 3], center_pose[:, :3, 3]], axis=0)  # (n+1+n, 3)
        rays_d_all = np.concatenate([
            rays_d,
            normalize(look_at_point[None, :] - avg_pose[:, :3, 3]),
            normalize(np.array(list(origin))[None, :] - center_pose[:, :3, 3])
        ])  # (n+1+n, 3)
        points_all = np.concatenate([look_at_point[None, :], np.array(list(origin))[None, :]], axis=0)

        file_path = osp.join(RESULT_DIR, 'recenter_at_real_avg_pose.png')
        draw_3d_components(
            c2w_all,
            cam_colors=cam_colors_all,
            points=points_all,
            rays=(cam_loc_all, rays_d_all),
            ray_colors=rays_colors_all,
            sphere_radius=self.radius,
            sphere_origin=origin,
            title='Random {} cam on sphere, get avg pose and recenter by real_avg_pose'.format(n_cam),
            save_path=file_path
        )

    @staticmethod
    def get_max_abs_error(a, b):
        return float((a - b).abs().max())


if __name__ == '__main__':
    unittest.main()
