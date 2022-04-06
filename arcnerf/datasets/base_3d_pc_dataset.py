# -*- coding: utf-8 -*-

import numpy as np

from .base_3d_dataset import Base3dDataset


class Base3dPCDataset(Base3dDataset):
    """Base 3d dataset with images, cameras and point cloud, based on base3dDataset.
       Mainly provide functions for point cloud adjustmens
    """

    def __init__(self, cfgs, data_dir, mode, transforms):
        """For any of 3d dataset, images/intrinsics/c2w are required. mask is optional"""
        super(Base3dPCDataset, self).__init__(cfgs, data_dir, mode, transforms)

    def get_sparse_point_cloud(self):
        """Get sparse point cloud. You should write it in child class if needed

        Returns:
            point_cloud: a dict. only 'pts' is required, 'color', 'vis' is optional
                - pts: (n_pts, 3), xyz points
                - color: (n_pts, 3), rgb color.
                - vis: (n_cam, n_pts), visibility in each cam.
        """
        raise NotImplementedError('You must have your point_cloud init function in child class...')

    def filter_point_cloud(self):
        """Filter point cloud in pc_radius """
        if hasattr(self.cfgs, 'pc_radius') and self.cfgs.pc_radius > 0:
            if 'pts' not in self.point_cloud:
                raise RuntimeError('Not pts in point_cloud, do not use this function...')

            pts_valid = np.linalg.norm(self.point_cloud['pts'], axis=-1) < self.cfgs.pc_radius
            self.point_cloud['pts'] = self.point_cloud['pts'][pts_valid, :]

            if 'color' in self.point_cloud:
                self.point_cloud['color'] = self.point_cloud['color'][pts_valid, :]

            if 'vis' in self.point_cloud:
                self.point_cloud['vis'] = self.point_cloud['vis'][:, pts_valid]

    def norm_cam_pose(self):
        """Normalize camera pose by scale_radius, point cloud as well"""
        assert len(self.cameras) > 0, 'Not camera in dataset, do not use this func'
        if hasattr(self.cfgs, 'scale_radius') and self.cfgs.scale_radius > 0:
            cam_norm_t = []
            for camera in self.cameras:
                cam_norm_t.append(camera.get_cam_pose_norm())
            max_cam_norm_t = max(cam_norm_t)

            for camera in self.cameras:
                camera.rescale_pose(scale=self.cfgs.scale_radius / (max_cam_norm_t * 1.1))

            # for point cloud adjustment
            self.point_cloud['pts'] *= (self.cfgs.scale_radius / (max_cam_norm_t * 1.1))
