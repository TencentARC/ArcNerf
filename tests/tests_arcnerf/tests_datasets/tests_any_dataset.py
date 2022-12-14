#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import cv2
import numpy as np
import torch

from arcnerf.datasets import get_dataset
from arcnerf.datasets.transform.augmentation import get_transforms
from arcnerf.geometry.poses import average_poses, generate_cam_pose_on_sphere
from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from arcnerf.geometry.volume import Volume
from arcnerf.models.fg_model import FgModel
from arcnerf.render.camera import PerspectiveCamera
from arcnerf.render.ray_helper import equal_sample, get_rays
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.cfgs_utils import get_value_from_cfgs_field, valid_key_in_cfgs
from common.utils.img_utils import img_to_uint8
from common.utils.torch_utils import np_wrapper, torch_to_np
from common.utils.video_utils import write_video
from common.visual import get_combine_colors
from common.visual.draw_cv2 import draw_vert_on_img
from . import setup_test_config

MODE = 'train'
RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfgs = setup_test_config()
        cls.dataset_type = getattr(cls.cfgs.dataset, MODE).type
        cls.dataset = cls.setup_dataset()
        cls.H, cls.W = int(cls.dataset[0]['H']), int(cls.dataset[0]['W'])
        cls.images = cls.load_images()
        cls.c2w, cls.intrinsic, cls.cameras = cls.get_cameras()
        cls.n_cam = cls.c2w.shape[0]
        radius = get_value_from_cfgs_field(cls.cfgs.model.rays, 'bounding_radius')
        cls.radius = radius if radius is not None else float(np.linalg.norm(cls.c2w[:, :3, 3], axis=-1).max(0))
        cls.get_inference_cfgs()
        cls.spec_result_dir = osp.abspath(osp.join(RESULT_DIR, cls.dataset_type, cls.dataset.get_identifier()))
        os.makedirs(cls.spec_result_dir, exist_ok=True)

    @classmethod
    def setup_dataset(cls):
        transforms, _ = get_transforms(getattr(cls.cfgs.dataset, MODE))
        dataset = get_dataset(cls.cfgs.dataset, cls.cfgs.dir.data_dir, None, MODE, transforms)

        return dataset

    @classmethod
    def load_images(cls):
        """img in bgr order"""
        imgs = []
        for data in cls.dataset:
            img = data['img'].clone().reshape(cls.H, cls.W, -1)
            img = img_to_uint8(torch_to_np(img))
            imgs.append(img)

        return imgs

    @classmethod
    def get_cameras(cls):
        intrinsic = cls.dataset.get_intrinsic(torch_tensor=False).astype(np.float32)  # (3, 3)
        c2w = cls.dataset.get_poses(torch_tensor=False, concat=True).astype(np.float32)  # (n, 4, 4)
        cameras = []
        for idx in range(c2w.shape[0]):
            cameras.append(PerspectiveCamera(intrinsic, c2w[idx], cls.W, cls.H))

        return c2w, intrinsic, cameras

    @classmethod
    def get_inference_cfgs(cls):
        infer_cfgs = get_value_from_cfgs_field(cls.cfgs, 'inference')

        if infer_cfgs is not None and valid_key_in_cfgs(infer_cfgs, 'render'):
            cls.parse_render(infer_cfgs.render)
        else:
            cls.render_type = ['circle', 'spiral']
            cls.render_n_cam = [30, 60]
            cls.render_radius = 3.0
            cls.render_u_start = 0.0
            cls.render_u_range = (0, 0.5)
            cls.render_v_ratio = 0.0
            cls.render_v_range = (-0.5, 0.0)
            cls.render_normal = (0.0, 1.0, 0.0)
            cls.render_n_rot = 3
            cls.render_reverse = False
            cls.render_fps = 5

        if infer_cfgs is not None and valid_key_in_cfgs(infer_cfgs, 'volume'):
            cls.parse_volume(infer_cfgs.volume)
            cls.volume = Volume(cls.n_grid, cls.origin, cls.side, cls.xyz_len)
            cls.volume_dict = {
                'grid_pts': torch_to_np(cls.volume.get_grid_pts()),
                'lines': cls.volume.get_dense_lines()
            }
        else:
            cls.volume = None
            cls.volume_dict = None

    @classmethod
    def parse_render(cls, render_cfgs):
        cls.render_type = get_value_from_cfgs_field(render_cfgs, 'type', ['circle', 'spiral'])
        cls.render_n_cam = get_value_from_cfgs_field(render_cfgs, 'n_cam', [30, 60])
        cls.render_radius = get_value_from_cfgs_field(render_cfgs, 'radius', 3.0)
        cls.render_u_start = get_value_from_cfgs_field(render_cfgs, 'u_start', 0.0)
        cls.render_u_range = tuple(get_value_from_cfgs_field(render_cfgs, 'u_range', [0, 0.5]))
        cls.render_v_ratio = get_value_from_cfgs_field(render_cfgs, 'v_ratio', 0.0)
        cls.render_v_range = tuple(get_value_from_cfgs_field(render_cfgs, 'v_range', [-0.5, 0.0]))
        cls.render_normal = tuple(get_value_from_cfgs_field(render_cfgs, 'normal', [0, 1.0, 0.0]))
        cls.render_n_rot = get_value_from_cfgs_field(render_cfgs, 'n_rot', 3)
        cls.render_reverse = get_value_from_cfgs_field(render_cfgs, 'reverse', False)
        cls.render_fps = get_value_from_cfgs_field(render_cfgs, 'fps', 5)

    @classmethod
    def parse_volume(cls, volume_cfgs):
        cls.n_grid = get_value_from_cfgs_field(volume_cfgs, 'n_grid', 4)
        cls.origin = tuple(get_value_from_cfgs_field(volume_cfgs, 'origin', [0.0, 0.0, 0.0]))
        cls.xyz_len = get_value_from_cfgs_field(volume_cfgs, 'xyz_len', None)
        if cls.xyz_len is None:
            cls.side = get_value_from_cfgs_field(volume_cfgs, 'side', 1.0)  # make sure volume exist
        else:
            cls.side = get_value_from_cfgs_field(volume_cfgs, 'side', None)

    def tests_get_dataset(self):
        self.assertIsInstance(self.dataset[0], dict)

    def tests_save_image(self):
        # rgb
        file_path = osp.join(self.spec_result_dir, 'img.png')
        img = self.dataset[0]['img'].reshape(self.H, self.W, -1)
        img = img_to_uint8(torch_to_np(img))
        cv2.imwrite(file_path, img)
        # mask
        if 'mask' in self.dataset[0]:
            file_path = osp.join(self.spec_result_dir, 'mask.png')
            mask = self.dataset[0]['mask'].reshape(self.H, self.W)
            mask = (torch_to_np(mask) * 255.0).astype(np.uint8)
            cv2.imwrite(file_path, mask)

    def tests_vis_cameras(self):
        origin = (0, 0, 0)
        # combine avg pose with different color
        avg_pose = average_poses(self.c2w)[None, :]  # (1, 4, 4)
        c2w = np.concatenate([self.c2w, avg_pose], axis=0)  # (n+1, 4, 4)
        cam_colors = get_combine_colors(['red', 'maroon'], [self.n_cam, 1])  # (n+1, 3)
        cam_loc = np.concatenate([self.c2w[:, :3, 3], avg_pose[:, :3, 3]])  # (n+1, 3)

        # rays_d is central ray from each camera
        rays_d = []
        center_index = np.array([[int(self.W / 2.0), int(self.H / 2.0)]])  # (1, 2)
        for i in range(self.c2w.shape[0]):
            ray_bundle = np_wrapper(get_rays, self.W, self.H, self.intrinsic, self.c2w[i], True, center_index)
            rays_d.append(ray_bundle[1])  # (1, 3)
        rays_d = np.concatenate(rays_d, axis=0)
        avg_rays_d = normalize(np.array(origin)[None, :] - avg_pose[:, :3, 3])  # (1, 3)
        rays_d = np.concatenate([rays_d, avg_rays_d], axis=0)  # (n+1, 3)
        rays_colors = get_combine_colors(['blue', 'navy'], [self.n_cam, 1])  # (n+1, 3)

        # max z as far
        z = np.linalg.norm(c2w[:, :3, 3], axis=-1).max()

        file_path = osp.join(self.spec_result_dir, '{}_vis_camera.png'.format(self.dataset_type))
        draw_3d_components(
            c2w,
            intrinsic=self.intrinsic,
            cam_colors=cam_colors,
            points=np.array(origin)[None, :],
            rays=(cam_loc, rays_d * z),
            ray_colors=rays_colors,
            sphere_radius=self.radius,
            sphere_origin=origin,
            title='{} Cam position'.format(self.dataset_type),
            save_path=file_path,
            plotly=True,
            plotly_html=True,
        )

    def tests_create_infer_cam_path(self):
        origin = (0, 0, 0)
        avg_pose = average_poses(self.c2w)[None, :]  # (1, 4, 4)

        for idx, mode in enumerate(self.render_type):
            file_path = osp.join(self.spec_result_dir, 'test_cam_mode_{}.png'.format(mode))
            c2w_test = generate_cam_pose_on_sphere(
                mode,
                self.render_radius,
                self.render_n_cam[idx],
                u_start=self.render_u_start,
                u_range=self.render_u_range,
                v_ratio=self.render_v_ratio,
                v_range=self.render_v_range,
                origin=origin,
                normal=self.render_normal,
                n_rot=self.render_n_rot,
                reverse=self.render_reverse,
                close=False  # just for test, should be true for actual visual
            )
            c2w = np.concatenate([c2w_test, avg_pose], axis=0)  # (n+1, 4, 4)
            cam_colors = get_combine_colors(['red', 'maroon'], [self.render_n_cam[idx], 1])  # (n+1, 3)
            cam_loc = np.concatenate([c2w_test[:, :3, 3], avg_pose[:, :3, 3]])  # (n+1, 3)
            rays_d = normalize(np.array(origin)[None, :] - cam_loc)  # (n+1, 3)
            rays_colors = get_combine_colors(['blue', 'navy'], [self.render_n_cam[idx], 1])  # (n+1, 3)
            draw_3d_components(
                c2w,
                intrinsic=self.intrinsic,
                cam_colors=cam_colors,
                points=np.array(origin)[None, :],
                rays=(cam_loc, rays_d),
                ray_colors=rays_colors,
                sphere_radius=self.radius,
                sphere_origin=origin,
                lines=[cam_loc[:self.render_n_cam[idx]]],
                title='Cam pos on sphere. Mode: {}'.format(mode),
                save_path=file_path,
                plotly=True,
                plotly_html=True
            )

    def tests_ray_points(self):
        n_rays_w, n_rays_h = 4, 3
        n_rays = n_rays_w * n_rays_h
        index = equal_sample(n_rays_w, n_rays_h, self.W, self.H)

        # custom chunk
        self.cfgs.model.chunk_rays = 1024
        self.cfgs.model.chunk_pts = 1024 * 32

        # change this range if you are not set cam in radius=3
        n_pts = 15

        # get combine bounds and rays
        rays_o = []
        rays_d = []
        bounds = []
        for idx in range(self.n_cam):
            ray_bundle = np_wrapper(get_rays, self.W, self.H, self.intrinsic, self.c2w[idx], False, index)[:2]
            rays_o.append(ray_bundle[0])  # (n_rays, 3)
            rays_d.append(ray_bundle[1])  # (n_rays, 3)
            if 'bounds' in self.dataset[idx]:  # (n_rays, 2)
                bounds.append(self.dataset[idx]['bounds'][:n_rays])

        rays_o = np.concatenate(rays_o, axis=0)  # (n_rays*n_cam, 3)
        rays_d = np.concatenate(rays_d, axis=0)  # (n_rays*n_cam, 3)
        bounds = None if len(bounds) == 0 else np.concatenate(bounds, axis=0)  # (n_rays*n_cam, 2)

        # use the foreground for sampling pts
        model = FgModel(self.cfgs)

        inputs = {
            'rays_o': torch.tensor(rays_o),
            'rays_d': torch.tensor(rays_d),
            'bounds': torch.tensor(bounds) if bounds is not None else None
        }
        near, far, _ = model.get_near_far_from_rays(inputs)
        zvals, _ = model.get_zvals_from_near_far(near, far, n_pts)  # (n_rays*n_cam, n_pts)
        zvals = torch_to_np(zvals)
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
            volume=self.volume_dict,
            sphere_radius=self.radius,
            sphere_origin=(0, 0, 0),
            title='Each cam sampled {} rays'.format(n_rays),
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

        title_with_setting = 'Each ray sampled {} points, z_range({:.1f}-{:.1f}).\n'.format(
            n_pts, zvals.min(), zvals.max()
        )
        file_path = osp.join(self.spec_result_dir, 'points_from_all_cams.png')
        draw_3d_components(
            self.c2w,
            intrinsic=self.intrinsic,
            cam_colors=cam_colors,
            points=points_all,
            point_colors=point_colors,
            point_size=5,
            volume=self.volume_dict,
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
            proj_imgs.append(draw_vert_on_img(self.images[idx].copy(), pts_pixels, color='green'))

        video_path = osp.join(self.spec_result_dir, 'reproj_pc.mp4')
        write_video(proj_imgs, video_path, fps=5)

    def tests_pc_plot3d(self):
        if 'pc' not in self.dataset[0] or self.dataset[0]['pc'] is None:
            return
        pc = self.dataset[0]['pc']
        if self.dataset[0]['img'].shape[0] != (self.H * self.W):  # Sample
            return

        pts = pc['pts']  # (n_pts, 3)
        pts_colors = pc['color'] if 'color' in pc else None

        if pts_colors is not None:
            self.assertEqual(pts_colors.shape, pts.shape)  # (n_pts, 3)

        file_path = osp.join(self.spec_result_dir, 'point_cloud_3d.png')
        draw_3d_components(
            c2w=self.c2w,
            intrinsic=self.intrinsic,
            points=pts,
            point_colors=pts_colors,
            point_size=5.0,
            volume=self.volume_dict,
            sphere_radius=self.radius,
            sphere_origin=(0, 0, 0),
            title='Cams with all point cloud',
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

        # single camera visual
        cam = self.c2w[:1, ...]
        index = np.array([[0, 0], [0, self.H - 1], [self.W - 1, 0], [self.W - 1, self.H - 1]])
        ray_bundle = self.cameras[0].get_rays(index=index, to_np=True)
        ray_colors = get_combine_colors(['blue', 'green', 'yellow', 'maroon'], [1] * 4)

        file_path = osp.join(self.spec_result_dir, 'single_cam_ray_pc.png')
        draw_3d_components(
            cam,
            intrinsic=self.intrinsic,
            points=pts,
            point_colors=pts_colors,
            point_size=5.0,
            rays=(ray_bundle[0], ray_bundle[1]),
            ray_colors=ray_colors,
            title='Cam ray at corner(lt: blue/lb: green/rt: yellow/rb: maroon)',
            save_path=file_path,
            plotly=True,
            plotly_html=True,
        )

    def tests_single_cam_ray(self):
        cam = self.c2w[:1, ...]
        index = np.array([[0, 0], [0, self.H - 1], [self.W - 1, 0], [self.W - 1, self.H - 1]])
        ray_bundle = self.cameras[0].get_rays(index=index, to_np=True)
        ray_colors = get_combine_colors(['blue', 'green', 'yellow', 'maroon'], [1] * 4)

        center_point = np.array([[0.0, 0.0, 0.0]])

        file_path = osp.join(self.spec_result_dir, 'single_cam_ray.png')
        draw_3d_components(
            cam,
            intrinsic=self.intrinsic,
            points=center_point,
            point_size=25.0,
            rays=(ray_bundle[0], ray_bundle[1]),
            ray_colors=ray_colors,
            title='Cam ray at corner(lt: blue/lb: green/rt: yellow/rb: maroon)',
            save_path=file_path,
            plotly=True,
            plotly_html=True,
        )


if __name__ == '__main__':
    unittest.main()
