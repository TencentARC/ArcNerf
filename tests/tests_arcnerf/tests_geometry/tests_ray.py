#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np

from arcnerf.geometry.poses import look_at, generate_cam_pose_on_sphere
from arcnerf.geometry.ray import (
    closest_point_on_ray, closest_point_to_rays, closest_point_to_two_rays, get_ray_points_by_zvals,
    sphere_ray_intersection
)
from arcnerf.geometry.transformation import normalize
from arcnerf.render.camera import PerspectiveCamera
from arcnerf.render.ray_helper import equal_sample, get_rays
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.torch_utils import np_wrapper
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
        return PerspectiveCamera(cls.intrinsic, cls.c2w, cls.W, cls.H)

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
        points_all = np.concatenate([np.array([self.origin]), points], axis=0)
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

    def get_ray_from_c2w(self, c2w, index):
        n_cam = c2w.shape[0]
        rays_o = []
        rays_d = []
        for idx in range(n_cam):
            ray = np_wrapper(get_rays, self.W, self.H, self.intrinsic, c2w[idx], index)
            rays_o.append(ray[0])
            rays_d.append(ray[1])
        rays = (np.concatenate(rays_o, axis=0), np.concatenate(rays_d, axis=0))
        n_rays = rays[0].shape[0]

        return rays, n_rays

    def tests_closest_point_on_ray(self):
        pts = np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=np.float32)
        # (N_rays, N_pts=2, 3)
        pts_closest, _ = np_wrapper(closest_point_on_ray, self.ray_bundle[0], self.ray_bundle[1], pts)
        points_all = np.concatenate([pts, pts_closest.transpose([1, 0, 2]).reshape(-1, 3)], axis=0)
        point_colors = get_combine_colors(['green', 'red', 'yellow'], [2, self.n_rays, self.n_rays])

        file_path = osp.join(RESULT_DIR, 'closest_point_on_ray.png')
        draw_3d_components(
            self.c2w[None, :],
            intrinsic=self.intrinsic,
            points=points_all,
            point_colors=point_colors,
            rays=(self.ray_bundle[0], self.z_max * self.ray_bundle[1]),
            ray_linewidth=0.5,
            title='{} point find closest point on {} rays'.format(pts.shape[0], self.n_rays),
            save_path=file_path,
        )

    def tests_closest_point_to_rays(self):
        n_cam = 10
        c2w = generate_cam_pose_on_sphere(
            mode='random', radius=self.radius, n_cam=n_cam, origin=self.origin, look_at_point=np.array(self.origin)
        ).astype(self.intrinsic.dtype)
        # central ray
        center_idx = np.array([[int(self.W / 2.0), int(self.H / 2.0)]])
        rays, n_rays = self.get_ray_from_c2w(c2w, center_idx)
        pts_closest, distance, zvals = np_wrapper(closest_point_to_rays, rays[0], rays[1])  # (1, 3), n, (n, 1)

        pts_on_rays = np_wrapper(get_ray_points_by_zvals, rays[0], rays[1], zvals)[:, 0, :]  # (n, 3)
        pts_all = np.concatenate([pts_closest, pts_on_rays])
        pts_colors = get_combine_colors(['green', 'red'], [1, n_cam])
        lines = [np.concatenate([pts_closest, pts_on_rays[idx:idx + 1]]) for idx in range(n_cam)]

        file_path = osp.join(RESULT_DIR, 'closest_point_to_rays(from_cam_center).png')
        draw_3d_components(
            c2w,
            intrinsic=self.intrinsic,
            points=pts_all,
            point_colors=pts_colors,
            rays=(rays[0], zvals * 1.2 * rays[1]),
            ray_linewidth=0.5,
            lines=lines,
            sphere_origin=self.origin,
            sphere_radius=self.radius,
            title='find closest point to {} rays'.format(n_rays),
            save_path=file_path
        )

        # (0, 0) ray from cam, no intersect at (0,0,0)
        left_top_idx = np.array([[0, 0]])
        rays, n_rays = self.get_ray_from_c2w(c2w, left_top_idx)
        pts_closest, distance, zvals = np_wrapper(closest_point_to_rays, rays[0], rays[1])  # (1, 3), n, (n, 1)

        pts_on_rays = np_wrapper(get_ray_points_by_zvals, rays[0], rays[1], zvals)[:, 0, :]  # (n, 3)
        pts_all = np.concatenate([pts_closest, pts_on_rays])
        pts_colors = get_combine_colors(['green', 'red'], [1, n_cam])
        lines = [np.concatenate([pts_closest, pts_on_rays[idx:idx + 1]]) for idx in range(n_cam)]

        file_path = osp.join(RESULT_DIR, 'closest_point_to_rays(from_(0,0)_of_cam).png')
        draw_3d_components(
            c2w,
            intrinsic=self.intrinsic,
            points=pts_all,
            point_colors=pts_colors,
            rays=(rays[0], zvals * 1.2 * rays[1]),
            ray_linewidth=0.5,
            lines=lines,
            sphere_origin=self.origin,
            sphere_radius=self.radius,
            title='find closest point to {} rays'.format(n_rays),
            save_path=file_path,
        )

    def tests_closest_point_to_two_rays(self):
        c2w = generate_cam_pose_on_sphere(
            mode='random', radius=self.radius, n_cam=2, origin=self.origin, look_at_point=np.array(self.origin)
        ).astype(self.intrinsic.dtype)

        # central ray
        center_idx = np.array([[int(self.W / 2.0), int(self.H / 2.0)]])
        rays, n_rays = self.get_ray_from_c2w(c2w, center_idx)  # (2, 3), (2, 3)
        pts_closest, distance, zvals = np_wrapper(closest_point_to_two_rays, rays[0], rays[1])  # (1, 3), 1, (2, 1)

        pts_on_rays = np_wrapper(get_ray_points_by_zvals, rays[0], rays[1], zvals)[:, 0, :]  # (2, 3)
        pts_all = np.concatenate([pts_closest, pts_on_rays])
        pts_colors = get_combine_colors(['green', 'red'], [1, 2])
        lines = [np.concatenate([pts_closest, pts_on_rays[0:1]]), np.concatenate([pts_closest, pts_on_rays[1:2]])]

        file_path = osp.join(RESULT_DIR, 'closest_point_to_two_rays(from _cam_center).png')
        draw_3d_components(
            c2w,
            intrinsic=self.intrinsic,
            points=pts_all,
            point_colors=pts_colors,
            rays=(rays[0], zvals * 1.2 * rays[1]),
            ray_linewidth=0.5,
            lines=lines,
            sphere_origin=self.origin,
            sphere_radius=self.radius,
            title='find closest point to {} rays'.format(n_rays),
            save_path=file_path
        )

        # (0, 0) ray from cam, no intersect at (0,0,0)
        left_top_idx = np.array([[0, 0]])
        rays, n_rays = self.get_ray_from_c2w(c2w, left_top_idx)
        pts_closest, distance, zvals = np_wrapper(closest_point_to_two_rays, rays[0], rays[1])  # (1, 3), 1, (2, 1)
        pts_on_rays = np_wrapper(get_ray_points_by_zvals, rays[0], rays[1], zvals)[:, 0, :]  # (2, 3)
        pts_all = np.concatenate([pts_closest, pts_on_rays])
        pts_colors = get_combine_colors(['green', 'red'], [1, 2])
        lines = [np.concatenate([pts_closest, pts_on_rays[0:1]]), np.concatenate([pts_closest, pts_on_rays[1:2]])]

        file_path = osp.join(RESULT_DIR, 'closest_point_to_two_rays(from_(0,0)_of_cam).png')
        draw_3d_components(
            c2w,
            intrinsic=self.intrinsic,
            points=pts_all,
            point_colors=pts_colors,
            rays=(rays[0], zvals * 1.2 * rays[1]),
            ray_linewidth=0.5,
            lines=lines,
            sphere_origin=self.origin,
            sphere_radius=self.radius,
            title='find closest point to {} rays'.format(n_rays),
            save_path=file_path
        )

        # central ray backward
        center_idx = np.array([[int(self.W / 2.0), int(self.H / 2.0)]])
        rays, n_rays = self.get_ray_from_c2w(c2w, center_idx)  # (2, 3), (2, 3)
        rays = list(rays)
        rays[1] = -1 * rays[1]
        pts_closest, distance, zvals = np_wrapper(closest_point_to_two_rays, rays[0], rays[1])  # (1, 3), 1, (2, 1)
        pts_on_rays = np_wrapper(get_ray_points_by_zvals, rays[0], rays[1], zvals)[:, 0, :]  # (2, 3)
        pts_all = np.concatenate([pts_closest, pts_on_rays])
        pts_colors = get_combine_colors(['green', 'red'], [1, 2])
        lines = [np.concatenate([pts_closest, pts_on_rays[0:1]]), np.concatenate([pts_closest, pts_on_rays[1:2]])]

        file_path = osp.join(RESULT_DIR, 'closest_point_to_two_rays_backward.png')
        draw_3d_components(
            c2w,
            intrinsic=self.intrinsic,
            points=pts_all,
            point_colors=pts_colors,
            rays=(rays[0], rays[1]),
            ray_linewidth=0.5,
            lines=lines,
            sphere_origin=self.origin,
            sphere_radius=self.radius,
            title='find closest point to {} rays'.format(n_rays),
            save_path=file_path
        )

    def tests_sphere_ray_intersection(self):
        # set sphere
        origin = (1, 1, 0)
        radius = 1.0

        # set rays
        rays_o = np.array([
            [1.5, 1, 0],  # inside
            [0.5, 1.5, 0],
            [0, 0, 0],  # outside
            [0.2, 0.2, 0],
            [-0.5, 0.5, 0],
            [-0.5, -0.5, 0],
            [0, 1, 0],  # on surface
            [0, 1, 0]
        ])
        rays_d = np.array([
            [1, 0, 0],  # 1 intersection, OC*D < 0
            [1, 0, 0],  # 1 intersection, OC*D > 0
            [1, 0, 0],  # 1 intersection(tangent）
            [1, 0.1, 0],  # 2 intersection
            [0, 1, 0],  # no intersection, similar direction to origin
            [-1, -1, 0],  # no intersection, opposition direction to origin
            [-1, 0, 0],  # one intersection, on surface outter ray
            [1, 10, 0]
        ])  # two intersection, on surface inner ray
        rays_d = normalize(rays_d)
        # get intersection
        near, far, pts, mask = np_wrapper(sphere_ray_intersection, rays_o, rays_d, origin, radius)
        pts = pts[mask, :].reshape(-1, 3)  # (n_valid_ray * 2, 3)

        rays_d[mask, :] *= far[mask, :] * 1.2

        blue_color = get_combine_colors(['blue'], [1])
        ray_colors = get_combine_colors(['red'], [rays_o.shape[0]])
        ray_colors[mask, :] = blue_color[0]

        file_path = osp.join(RESULT_DIR, 'sphere_ray_intersection.png')
        draw_3d_components(
            points=pts,
            rays=(rays_o, rays_d),
            ray_linewidth=0.5,
            ray_colors=ray_colors,
            sphere_origin=origin,
            sphere_radius=radius,
            title='sphere ray intersection(ray in red no intersection)',
            save_path=file_path
        )

    @staticmethod
    def get_max_abs_error(a, b):
        return float((a - b).abs().max())


if __name__ == '__main__':
    unittest.main()
