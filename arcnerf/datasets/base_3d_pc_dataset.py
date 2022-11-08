# -*- coding: utf-8 -*-

import numpy as np

from arcnerf.geometry.poses import center_poses
from arcnerf.geometry.projection import world_to_cam
from arcnerf.geometry.transformation import rotate_points
from common.utils.cfgs_utils import valid_key_in_cfgs
from common.utils.torch_utils import np_wrapper
from .base_3d_dataset import Base3dDataset


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

    def get_holdout_samples(self, holdout_index):
        """Get the holdout split for images/camera, etc
        holdout pts_vis as well
        """
        super().get_holdout_samples(holdout_index)
        if 'vis' in self.point_cloud:
            self.point_cloud['vis'] = self.point_cloud['vis'][holdout_index, :]

    def get_holdout_samples_with_list(self, holdout_index, img_list, mask_list=None):
        """Get the holdout split for images/camera, etc, img are not read but given list
        holdout pts_vis as well
        """
        img_list, mask_list = super().get_holdout_samples_with_list(holdout_index, img_list, mask_list)
        if 'vis' in self.point_cloud:
            self.point_cloud['vis'] = self.point_cloud['vis'][holdout_index, :]

        return img_list, mask_list

    def skip_samples(self):
        """For any mode, you can skip the samples in order.
        Skip pts_vis as well
        """
        super().skip_samples()
        if self.skip > 1:
            if 'vis' in self.point_cloud:
                self.point_cloud['vis'] = self.point_cloud['vis'][::self.skip, :]

    def skip_samples_with_list(self, img_list, mask_list=None):
        """Do not real image at the beginning, skip the img_list/mask_list for further loading.
        Skip pts_vis as well
        """
        img_list, mask_list = super().skip_samples_with_list(img_list, mask_list)
        if self.skip > 1:
            if 'vis' in self.point_cloud:
                self.point_cloud['vis'] = self.point_cloud['vis'][::self.skip, :]

        return img_list, mask_list

    def keep_eval_samples(self):
        """For eval model, only keep a small number of samples. Which are closer to the avg pose
         It should be done before precache_rays in child class to avoid full precache.
         """
        ind = super().keep_eval_samples()
        if ind is not None:
            if 'vis' in self.point_cloud:
                self.point_cloud['vis'] = self.point_cloud['vis'][ind, :]

    def filter_point_cloud(self):
        """Filter point cloud in pc_radius, it is in scale after cam normalization """
        # TODO: Better filter to remove outlier points should be applied
        if valid_key_in_cfgs(self.cfgs, 'pc_radius') and self.cfgs.pc_radius > 0:
            pts_valid = np.linalg.norm(self.point_cloud['pts'], axis=-1) < (self.cfgs.pc_radius / 1.05)
            self.point_cloud['pts'] = self.point_cloud['pts'][pts_valid, :]

            if 'color' in self.point_cloud:
                self.point_cloud['color'] = self.point_cloud['color'][pts_valid, :]

            if 'vis' in self.point_cloud:
                self.point_cloud['vis'] = self.point_cloud['vis'][:, pts_valid]

    def adjust_cam_translation(self):
        """Adjust c2w translation by some offset. This could make the obj in the coord center"""
        assert len(self.cameras) > 0, 'Not camera in dataset, do not use this func'

        if valid_key_in_cfgs(self.cfgs, 'cam_t_offset') and len(self.cfgs.cam_t_offset) == 3:
            offset = np.array(self.cfgs.cam_t_offset)  # (3,)
            for camera in self.cameras:
                camera.adjust_translation(-offset)  # always negative direction

            # adjust point cloud as well
            self.point_cloud['pts'] -= offset[None]

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

    def center_cam_poses_by_view_dirs(self):
        """Recenter camera pose by setting the common view point center at (0,0,0)
        The common view point is the closest point to all rays.
        """
        view_point_mean = super().center_cam_poses_by_view_dirs()
        if view_point_mean is not None:
            self.point_cloud['pts'] -= view_point_mean

    def norm_cam_pose(self):
        """Normalize camera pose by scale_radius, point cloud as well"""
        max_cam_norm_t = super().norm_cam_pose()
        # for point cloud adjustment
        self.point_cloud['pts'] *= (self.cfgs.scale_radius / (max_cam_norm_t * 1.05))

    def align_cam_horizontal(self):
        """Align all camera direction and position to up.
        Use it only when camera are not horizontally around the object
        """
        rot_mat = super().align_cam_horizontal()
        if rot_mat is not None:
            rot_mat_pts = rot_mat.copy().astype(self.point_cloud['pts'].dtype)[None]
            self.point_cloud['pts'] = np_wrapper(rotate_points, self.point_cloud['pts'][None], rot_mat_pts)[0]

    def get_bounds_from_pc(self, extend_factor=0.05):
        """Get bounds from pc projected by each cam.
         near-far by pts_cam and adjust by extend_factor, in case pc does not cover all range.
        """
        bounds = []
        for idx in range(len(self.cameras)):
            cam_cpu = self.cameras[idx].get_pose(w2c=True).cpu()
            pts_reproj = np_wrapper(world_to_cam, self.point_cloud['pts'][None, ...], cam_cpu[None, ...])[0]
            near, far = pts_reproj[:, -1].min(), pts_reproj[:, -1].max()
            if extend_factor > 0:
                near_far_dist = far - near
                near -= extend_factor * near_far_dist
                far += extend_factor * near_far_dist
            near, far = np.clip(near, 0.0, None), np.clip(far, 0.0, None)
            bound = np.array([near, far], dtype=near.dtype)
            bounds.append(bound)

        return bounds
