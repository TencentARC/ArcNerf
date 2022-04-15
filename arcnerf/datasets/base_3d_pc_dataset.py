# -*- coding: utf-8 -*-

import numpy as np

from .base_3d_dataset import Base3dDataset
from arcnerf.geometry.poses import center_poses
from arcnerf.geometry.ray import closest_point_to_rays
from common.utils.torch_utils import np_wrapper


class Base3dPCDataset(Base3dDataset):
    """Base 3d dataset with images, cameras and point cloud, based on base3dDataset.
       Mainly provide functions for point cloud adjustmens
    """

    def __init__(self, cfgs, data_dir, mode, transforms):
        """For any of 3d dataset, images/intrinsics/c2w are required. mask is optional"""
        super(Base3dPCDataset, self).__init__(cfgs, data_dir, mode, transforms)

    def get_sparse_point_cloud(self, dtype=np.float32):
        """Get sparse point cloud. You should write it in child class if needed
        For the dtype, it should matched the cam's params type(which use torch.float32 as default)

        Returns:
            point_cloud: a dict. only 'pts' is required, 'color', 'vis' is optional
                - pts: (n_pts, 3), xyz points
                - color: (n_pts, 3), rgb color. (0~1) float point
                - vis: (n_cam, n_pts), visibility in each cam.
            dtype: type of each component, by default np.float32
        """
        raise NotImplementedError('You must have your point_cloud init function in child class...')

    def get_identifier(self):
        """string identifier of a dataset like scan_id/scene_name"""
        return self.identifier

    def keep_eval_samples(self):
        """For eval model, only keep a small number of samples. Which are closer to the avg pose
         It should be done before precache_rays in child class to avoid full precache.
         """
        if self.eval_max_sample is not None:
            n_imgs = min(self.eval_max_sample, self.n_imgs)
            self.n_imgs = n_imgs
            ind = self.find_closest_cam_ind(n_imgs)
            self.images = [self.images[i] for i in ind]
            self.cameras = [self.cameras[i] for i in ind]
            self.masks = [self.masks[i] for i in ind] if len(self.masks) > 0 else []
            self.bounds = [self.bounds[i] for i in ind] if len(self.bounds) > 0 else []
            if 'vis' in self.point_cloud:
                self.point_cloud['vis'] = self.point_cloud['vis'][ind, :]

    def filter_point_cloud(self):
        """Filter point cloud in pc_radius, it is in scale after cam normalization """
        # TODO: Better filter to remove outlier points should be applied
        if hasattr(self.cfgs, 'pc_radius') and self.cfgs.pc_radius > 0:
            pts_valid = np.linalg.norm(self.point_cloud['pts'], axis=-1) < (self.cfgs.pc_radius / 1.05)
            self.point_cloud['pts'] = self.point_cloud['pts'][pts_valid, :]

            if 'color' in self.point_cloud:
                self.point_cloud['color'] = self.point_cloud['color'][pts_valid, :]

            if 'vis' in self.point_cloud:
                self.point_cloud['vis'] = self.point_cloud['vis'][:, pts_valid]

    def center_cam_poses_by_pc_mean(self):
        """Recenter camera pose by recenter the point_cloud center as (0,0,0)
        You should filter noisy point cloud that not belong to the main object
        """
        c2ws = self.get_poses(torch_tensor=False, concat=True)
        pts_mean = self.point_cloud['pts'].mean(0)
        center_c2w = center_poses(c2ws, pts_mean)
        for idx in range(len(self.cameras)):
            self.cameras[idx].reset_pose(center_c2w[idx])

        # adjust point cloud as well
        self.point_cloud['pts'] -= pts_mean[None]

    def center_cam_poses_by_view_dir(self):
        """Recenter camera pose by setting the common view point center at (0,0,0)
        The common view point is the closest point to all rays.
        """
        assert len(self.cameras) > 0, 'Not camera in dataset, do not use this func'
        c2ws = self.get_poses(torch_tensor=False, concat=True)
        # use ray from image center to represent cam view dir
        center_idx = np.array([[int(self.W / 2.0), int(self.H / 2.0)]])
        rays_o = []
        rays_d = []
        for idx in range(len(self.cameras)):
            ray = self.cameras[idx].get_rays(index=center_idx, to_np=True)
            rays_o.append(ray[0])
            rays_d.append(ray[1])
        rays = (np.concatenate(rays_o, axis=0), np.concatenate(rays_d, axis=0))
        # calculate mean view point
        view_point_mean, _, _ = np_wrapper(closest_point_to_rays, rays[0], rays[1])  # (1, 3)
        center_c2w = center_poses(c2ws, view_point_mean[0])
        for idx in range(len(self.cameras)):
            self.cameras[idx].reset_pose(center_c2w[idx])

        # if point cloud exist, also adjust it
        self.point_cloud['pts'] -= view_point_mean

    def norm_cam_pose(self):
        """Normalize camera pose by scale_radius, point cloud as well"""
        assert len(self.cameras) > 0, 'Not camera in dataset, do not use this func'
        if hasattr(self.cfgs, 'scale_radius') and self.cfgs.scale_radius > 0:
            cam_norm_t = []
            for camera in self.cameras:
                cam_norm_t.append(camera.get_cam_pose_norm())
            max_cam_norm_t = max(cam_norm_t)

            for camera in self.cameras:
                camera.rescale_pose(scale=self.cfgs.scale_radius / (max_cam_norm_t * 1.05))

            # for point cloud adjustment
            self.point_cloud['pts'] *= (self.cfgs.scale_radius / (max_cam_norm_t * 1.05))

    def get_bounds_from_pc(self, extend_factor=0.05):
        """Get bounds from pc projected by each cam.
         near-far by pts_cam and adjust by extend_factor, in case pc does not cover all range.
        """
        bounds = []
        for idx in range(len(self.cameras)):
            pts_reproj = np_wrapper(self.cameras[idx].proj_world_to_cam, self.point_cloud['pts'])
            near, far = pts_reproj[:, -1].min(), pts_reproj[:, -1].max()
            if extend_factor > 0:
                near_far_dist = far - near
                near -= extend_factor * near_far_dist
                far += extend_factor * near_far_dist
            near, far = np.clip(near, 0.0, None), np.clip(far, 0.0, None)
            bound = np.array([near, far], dtype=near.dtype)
            bounds.append(bound)

        return bounds
