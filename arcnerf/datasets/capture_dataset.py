# -*- coding: utf-8 -*-

import glob
import os.path as osp

import numpy as np

from .base_3d_pc_dataset import Base3dPCDataset
from ..datasets import DATASET_REGISTRY
from arcnerf.geometry.poses import invert_poses
from arcnerf.render.camera import PerspectiveCamera


@DATASET_REGISTRY.register()
class Capture(Base3dPCDataset):
    """A dataset class for self-capture images with colmap pose estimation"""

    def __init__(self, cfgs, data_dir, mode, transforms):
        super(Capture, self).__init__(cfgs, data_dir, mode, transforms)

        # real capture dataset with scene_name
        self.data_spec_dir = osp.join(self.data_dir, 'Capture', self.cfgs.scene_name)
        self.identifier = self.cfgs.scene_name

        # get image
        img_list, self.n_imgs = self.get_image_list()
        self.images = self.read_image_list(img_list)
        self.H, self.W = self.images[0].shape[:2]

        # get cameras
        self.cam_file = osp.join(self.data_spec_dir, 'poses_bounds.npy')
        assert osp.exists(self.cam_file), 'Camera file {} not exist...Please run colmap first...'.format(self.cam_file)
        self.poses = np.load(self.cam_file, allow_pickle=True).item()
        self.cameras = self.read_cameras()

        # get pointcloud
        self.point_cloud = self.get_sparse_point_cloud()

        # roughly center the cameras by common view point
        self.center_cam_poses_by_view_dir()
        # norm camera_pose to restrict pc range
        self.norm_cam_pose()
        # filter point outside sphere
        self.filter_point_cloud()
        # recenter the cameras by remaining point cloud
        self.center_cam_poses_by_pc_mean()
        # # re-norm again
        self.norm_cam_pose()

        # rescale image, call from parent class
        self.rescale_img_and_pose()

        # precache_all rays
        self.ray_bundles = None
        self.precache = self.cfgs.precache

        if self.precache:
            self.precache_ray()

    def get_image_list(self):
        """Get image list"""
        img_dir = osp.join(self.data_spec_dir, 'images')
        img_list = sorted(glob.glob(img_dir + '/*.png'))

        n_imgs = len(img_list)
        assert n_imgs > 0, 'No image exists in {}'.format(img_dir)

        return img_list, n_imgs

    def read_cameras(self):
        """Read camera from pose file"""
        assert self.poses['h'] == self.H and self.poses['w'] == self.W,\
            'Cam poses not match image size...image {}/{} - cam {}/{}'.format(self.W, self.H,
                                                                              self.poses['w'], self.poses['h'])

        w2c = np.concatenate([self.poses['R'], self.poses['T']], axis=-1)  # (N, 3, 4)
        bottom = np.repeat(np.array([0, 0, 0, 1.]).reshape([1, 4])[None, ...], w2c.shape[0], axis=0)  # (N, 1, 4)
        w2c = np.concatenate([w2c, bottom], axis=1)  # (N, 4, 4)
        c2w = invert_poses(w2c)
        intrinsic = self.get_colmap_intrinsic()

        cameras = []
        for idx in range(self.n_imgs):  # read only first n_imgs
            cameras.append(PerspectiveCamera(intrinsic=intrinsic, c2w=c2w[idx], W=self.W, H=self.H))

        return cameras

    def get_colmap_intrinsic(self):
        """Get intrinsic (3, 3) from pose file"""
        cam_type = self.poses['cam_type']
        cam_params = self.poses['cam_params']
        if cam_type == 'SIMPLE_RADIAL':  # f, cx, cy, k. Ignore k for simplicity
            intrinsic = np.eye(3)
            intrinsic[0, 0] = cam_params[0]
            intrinsic[1, 1] = cam_params[0]
            intrinsic[0, 2] = cam_params[1]
            intrinsic[1, 2] = cam_params[2]
        else:
            raise NotImplementedError('Not support cam mode {} from colmap reconstruction yet...'.format(cam_type))

        return intrinsic

    def get_sparse_point_cloud(self):
        """Get sparse point cloud as the point cloud. color should be normed in (0,1)"""
        pc = {
            'pts': self.poses['pts'],
            'color': self.poses['rgb'].astype(np.float64) / 255.0,
            'vis': self.poses['vis'][:self.n_imgs]
        }

        return pc
