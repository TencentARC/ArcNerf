#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np
import cv2

from . import setup_test_config
from arcnerf.datasets import get_dataset
from arcnerf.datasets.transform.augmentation import get_transforms
from arcnerf.geometry.poses import average_poses, generate_cam_pose_on_sphere
from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.sphere import get_uv_from_pos
from arcnerf.geometry.transformation import normalize
from arcnerf.render.camera import PerspectiveCamera
from arcnerf.render.ray_helper import equal_sample, get_rays, get_near_far_from_rays, get_zvals_from_near_far
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.torch_utils import np_wrapper
from common.utils.video_utils import write_video
from common.visual import get_combine_colors
from common.visual.draw_cv2 import draw_vert_on_img

MODE = 'train'
RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfgs = setup_test_config()
        cls.dataset_type = getattr(cls.cfgs.dataset, MODE).type
        cls.dataset = cls.setup_dataset()
        cls.images = cls.load_images()
        cls.H, cls.W = int(cls.dataset[0]['H']), int(cls.dataset[0]['W'])
        cls.c2w, cls.intrinsic, cls.cameras = cls.get_cameras()
        cls.n_cam = cls.c2w.shape[0]
        radius = get_value_from_cfgs_field(cls.cfgs.model.rays, 'bounding_radius')
        cls.radius = radius if radius is not None else np.linalg.norm(cls.c2w[:, :3, 3], axis=-1).max(0)
        cls.spec_result_dir = osp.abspath(osp.join(RESULT_DIR, cls.dataset_type, cls.dataset.get_identifier()))
        os.makedirs(cls.spec_result_dir, exist_ok=True)

    @classmethod
    def setup_dataset(cls):
        transforms, _ = get_transforms(getattr(cls.cfgs.dataset, MODE))
        dataset = get_dataset(cls.cfgs.dataset, cls.cfgs.dir.data_dir, None, MODE, transforms)

        return dataset

    @classmethod
    def load_images(cls):
        img_list, _ = cls.dataset.get_image_list()
        imgs = [cv2.imread(path) for path in img_list]

        return imgs

    @classmethod
    def get_cameras(cls):
        intrinsic = cls.dataset.get_intrinsic(torch_tensor=False).astype(np.float32)  # (3, 3)
        c2w = cls.dataset.get_poses(torch_tensor=False, concat=True).astype(np.float32)  # (n, 4, 4)
        cameras = []
        for idx in range(c2w.shape[0]):
            cameras.append(PerspectiveCamera(intrinsic, c2w[idx], cls.W, cls.H))

        return c2w, intrinsic, cameras

    def tests_get_dataset(self):
        self.assertIsInstance(self.dataset[0], dict)

    def tests_vis_cameras(self):
        origin = (0, 0, 0)
        # combine avg pose with different color
        avg_pose = average_poses(self.c2w)[None, :]  # (1, 4, 4)
        c2w = np.concatenate([self.c2w, avg_pose], axis=0)  # (n+1, 4, 4)
        cam_colors = get_combine_colors(['red', 'maroon'], [self.n_cam, 1])  # (n+1, 3)
        cam_loc = np.concatenate([self.c2w[:, :3, 3], avg_pose[:, :3, 3]])  # (n+1, 3)
        rays_d = normalize(np.array(origin)[None, :] - cam_loc)  # (n+1, 3)
        rays_colors = get_combine_colors(['blue', 'navy'], [self.n_cam, 1])  # (n+1, 3)

        file_path = osp.join(self.spec_result_dir, '{}_vis_camera.png'.format(self.dataset_type))
        draw_3d_components(
            c2w,
            intrinsic=self.intrinsic,
            cam_colors=cam_colors,
            points=np.array(origin)[None, :],
            rays=(cam_loc, rays_d),
            ray_colors=rays_colors,
            sphere_radius=self.radius,
            sphere_origin=origin,
            title='{} Cam position'.format(self.dataset_type),
            save_path=file_path,
        )

    def tests_create_test_cam_path(self):
        n_test_cam = 25
        origin = (0, 0, 0)
        avg_pose = average_poses(self.c2w)[None, :]  # (1, 4, 4)

        u_start, v_ratio, _ = get_uv_from_pos(avg_pose[0, :3, 3], origin, self.radius)
        _, v_min, _ = get_uv_from_pos(self.c2w[np.argmin(self.c2w[:, 1, 3]), :3, 3], origin, self.radius)
        _, v_max, _ = get_uv_from_pos(self.c2w[np.argmax(self.c2w[:, 1, 3]), :3, 3], origin, self.radius)

        modes = ['circle', 'spiral']
        for mode in modes:
            file_path = osp.join(self.spec_result_dir, 'test_cam_mode_{}.png'.format(mode))
            c2w_test = generate_cam_pose_on_sphere(
                mode,
                self.radius,
                n_test_cam,
                u_start=u_start,
                v_ratio=v_ratio,
                v_range=(v_min, v_max),
                origin=origin,
                n_rot=3,
                close=False  # just for test, should be true for actual visual
            )
            c2w = np.concatenate([c2w_test, avg_pose], axis=0)  # (n+1, 4, 4)
            cam_colors = get_combine_colors(['red', 'maroon'], [n_test_cam, 1])  # (n+1, 3)
            cam_loc = np.concatenate([c2w_test[:, :3, 3], avg_pose[:, :3, 3]])  # (n+1, 3)
            rays_d = normalize(np.array(origin)[None, :] - cam_loc)  # (n+1, 3)
            rays_colors = get_combine_colors(['blue', 'navy'], [n_test_cam, 1])  # (n+1, 3)
            draw_3d_components(
                c2w,
                intrinsic=self.intrinsic,
                cam_colors=cam_colors,
                points=np.array(origin)[None, :],
                rays=(cam_loc, rays_d),
                ray_colors=rays_colors,
                sphere_radius=self.radius,
                sphere_origin=origin,
                lines=[cam_loc[:n_test_cam]],
                title='Cam pos on sphere. Mode: {}'.format(mode),
                save_path=file_path
            )

    def tests_ray_points(self):
        n_rays_w, n_rays_h = 8, 6
        n_rays = n_rays_w * n_rays_h
        index = equal_sample(n_rays_w, n_rays_h, self.W, self.H)

        # near/far from config
        bounding_radius = get_value_from_cfgs_field(self.cfgs.model.rays, 'bounding_radius')
        near = get_value_from_cfgs_field(self.cfgs.model.rays, 'near')
        far = get_value_from_cfgs_field(self.cfgs.model.rays, 'far')

        # change this range if you are not set cam in radius=3
        n_pts = 15

        # get combine bounds and rays
        rays_o = []
        rays_d = []
        bounds = []
        for idx in range(self.n_cam):
            ray_bundle = np_wrapper(get_rays, self.W, self.H, self.intrinsic, self.c2w[idx], index)[:2]  # (n_rays, 3)
            rays_o.append(ray_bundle[0])
            rays_d.append(ray_bundle[1])
            if 'bounds' in self.dataset[idx]:  # (n_rays, 2)
                bounds.append(self.dataset[idx]['bounds'][:n_rays])

        rays_o = np.concatenate(rays_o, axis=0)  # (n_rays*n_cam, 3)
        rays_d = np.concatenate(rays_d, axis=0)  # (n_rays*n_cam, 3)
        bounds = None if len(bounds) == 0 else np.concatenate(bounds, axis=0)  # (n_rays*n_cam, 2)

        near_all, far_all = np_wrapper(get_near_far_from_rays, rays_o, rays_d, bounds, near, far, bounding_radius)
        zvals = np_wrapper(get_zvals_from_near_far, near_all, far_all, n_pts)  # (n_rays*n_cam, n_pts)
        pts = np_wrapper(get_ray_points_by_zvals, rays_o, rays_d, zvals)  # (n_rays*n_cam, n_pts, 3)
        points = pts.reshape(-1, 3)  # (n_rays*n_cam*n_pts, 3)
        self.assertEqual(points.shape, (self.n_cam * n_rays * n_pts, 3))
        self.assertEqual(rays_o.shape, (self.n_cam * n_rays, 3))
        self.assertEqual(rays_d.shape, (self.n_cam * n_rays, 3))

        cam_colors = get_combine_colors(['blue'], [self.c2w.shape[0]])

        ray_colors = get_combine_colors(['sky_blue'], [self.n_cam * n_rays])
        points_with_cam = np.concatenate([np.array([0, 0, 0])[None, :], self.c2w[:, :3, 3]], axis=0)
        point_colors_with_cam = get_combine_colors(['green', 'red'], [1, self.c2w.shape[0]])

        points_all = np.concatenate([np.array([0, 0, 0])[None, :], points], axis=0)
        point_colors = get_combine_colors(['green', 'red'], [1, points.shape[0]])
        if 'pc' in self.dataset[0] and self.dataset[0]['pc'] is not None:
            if self.dataset[0]['img'].shape[0] == (self.H * self.W):  # Sample
                pc = self.dataset[0]['pc']
                pts = pc['pts']  # (n_pts, 3)
                if 'color' in pc:  # only show
                    # add to pts
                    points_with_cam = np.concatenate([points_with_cam, pts], axis=0)
                    point_colors_with_cam = np.concatenate([point_colors_with_cam, pc['color']], axis=0)
                    points_all = np.concatenate([points_all, pts], axis=0)
                    point_colors = np.concatenate([point_colors, pc['color']], axis=0)

        file_path = osp.join(self.spec_result_dir, 'rays_from_all_cams.png')
        draw_3d_components(
            points=points_with_cam,
            point_size=1.0,
            point_colors=point_colors_with_cam,
            rays=(rays_o, zvals[:, -1][:, None] * rays_d),
            ray_colors=ray_colors,
            ray_linewidth=0.5,
            sphere_radius=self.radius,
            sphere_origin=(0, 0, 0),
            title='Each cam sampled {} rays'.format(n_rays),
            save_path=file_path,
        )

        title_with_setting = 'Each ray sampled {} points, z_range({:.1f}-{:.1f}).\n'.format(
            n_pts, zvals.min(), zvals.max()
        )
        title_with_setting += 'Settings - bounds:{}/near:{}/far:{}/bounding_radius:{}'.format(
            bounds is not None, near, far, bounding_radius
        )
        file_path = osp.join(self.spec_result_dir, 'points_from_all_cams.png')
        draw_3d_components(
            self.c2w,
            intrinsic=self.intrinsic,
            cam_colors=cam_colors,
            points=points_all,
            point_colors=point_colors,
            point_size=5,
            sphere_radius=self.radius,
            sphere_origin=(0, 0, 0),
            title=title_with_setting,
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

    def tests_pc_reproject(self):
        if 'pc' not in self.dataset[0] or self.dataset[0]['pc'] is None:
            return
        pc = self.dataset[0]['pc']
        if self.dataset[0]['img'].shape[0] != (self.H * self.W):  # Sample
            return

        pts = pc['pts']  # (n_pts, 3)
        pts_vis = pc['vis'] if 'vis' in pc else None
        if pts_vis is not None:
            self.assertEqual(pts_vis.shape[0], self.n_cam)  # (n_cam, n_pts)

        proj_imgs = []
        for idx in range(self.n_cam):
            pts_pixels = np_wrapper(self.cameras[idx].proj_world_to_pixel, pts)
            pts_vis_cam = pts_vis[idx, :]
            pts_pixels = pts_pixels if pts_vis is None else pts_pixels[pts_vis_cam == 1, :]

            proj_imgs.append(draw_vert_on_img(self.images[idx], pts_pixels, color='green'))

        video_path = osp.join(self.spec_result_dir, 'reproj_pc.mp4')
        write_video(proj_imgs, video_path, fps=5)

    def tests_pc_plot3d(self):
        if 'pc' not in self.dataset[0] or self.dataset[0]['pc'] is None:
            return
        pc = self.dataset[0]['pc']
        if self.dataset[0]['img'].shape[0] != (self.H * self.W):  # Sample
            return

        pts = pc['pts']  # (n_pts, 3)
        pts_color = pc['color'] if 'color' in pc else None

        if pts_color is not None:
            self.assertEqual(pts_color.shape, pts.shape)  # (n_pts, 3)

        file_path = osp.join(self.spec_result_dir, 'point_cloud_3d.png')
        draw_3d_components(
            c2w=self.c2w,
            intrinsic=self.intrinsic,
            points=pts,
            point_colors=pts_color,
            point_size=1.0,
            sphere_radius=self.radius,
            sphere_origin=(0, 0, 0),
            title='Cams with all point cloud',
            save_path=file_path
        )

        # single camera visual
        cam = self.c2w[:1, ...]
        index = np.array([[0, 0]])
        ray_bundle = self.cameras[0].get_rays(index=index, to_np=True)

        file_path = osp.join(self.spec_result_dir, 'single_cam_ray_pc.png')
        draw_3d_components(
            cam,
            intrinsic=self.intrinsic,
            points=pts,
            point_colors=pts_color,
            point_size=1.0,
            rays=(ray_bundle[0], ray_bundle[1]),
            title='Cam ray at (0,0).',
            save_path=file_path
        )


if __name__ == '__main__':
    unittest.main()
