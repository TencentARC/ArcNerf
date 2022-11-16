# -*- coding: utf-8 -*-

import glob
import os.path as osp

import numpy as np

from arcnerf.render.camera import PerspectiveCamera
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import DATASET_REGISTRY
from .base_3d_dataset import Base3dDataset


@DATASET_REGISTRY.register()
class TanksAndTemples(Base3dDataset):
    """TanksAndTemples Dataset. We used the version processed by nerf++ (https://github.com/Kai-46/nerfplusplus)
    which contains 4 scenes (Truck, M60, Train, Playground)
    The official link is https://www.tanksandtemples.org/, but it does not contains intrinsic and need further optim.
    """

    def __init__(self, cfgs, data_dir, mode, transforms):
        super(TanksAndTemples, self).__init__(cfgs, data_dir, mode, transforms)

        # real capture dataset with scene_name
        scene_dir = 'tat_{}_{}'.format(self.convert_scene(self.cfgs.scene_name), self.cfgs.scene_name)
        self.data_spec_dir = osp.join(self.data_dir, 'TanksAndTemples', scene_dir)
        self.identifier = self.cfgs.scene_name

        # get image
        img_list, self.n_imgs = self.get_image_list(mode)
        self.images = self.read_image_list(img_list)
        self.H, self.W = self.read_image_list(img_list[:1])[0].shape[:2]

        # load all camera together in all split for consistent camera normalization
        self.cameras, cam_split_idx = self.read_cameras_by_mode(mode)  # get the index for final selection
        for cam in self.cameras:
            cam.set_device(self.device)

        # handle the camera in all split to make consistent
        # norm camera_pose to restrict pc range
        self.norm_cam_pose()

        # keep only the camera in certain split
        self.cameras = [self.cameras[idx] for idx in cam_split_idx]
        assert self.n_imgs == len(self.cameras), 'Camera num not match the image number'

        # skip image and keep less samples
        self.skip_samples()
        # keep close-to-mean samples if set
        self.keep_eval_samples()

        # rescale image, call from parent class
        self.rescale_img_and_pose()

        # precache_all rays
        self.ray_bundles = None
        self.precache = get_value_from_cfgs_field(self.cfgs, 'precache', False)

        if self.precache:
            self.precache_ray()

    @staticmethod
    def convert_scene(scene_name):
        """Convert scene name to kind"""
        if scene_name == 'Truck':
            return 'training'
        else:
            return 'intermediate'

    @staticmethod
    def convert_mode(mode):
        """Convert mode train/val/eval to dataset name"""
        if mode == 'train':
            return 'train'
        elif mode == 'val' or mode == 'eval':  # bot read test
            return 'test'
        else:
            raise NotImplementedError('Not such mode {}...'.format(mode))

    def get_image_list(self, mode=None):
        """Get image list."""
        img_dir = osp.join(self.data_spec_dir, self.convert_mode(mode), 'rgb')
        img_list = sorted(glob.glob(img_dir + '/*.png'))

        n_imgs = len(img_list)
        assert n_imgs > 0, 'No image exists in {}'.format(img_dir)

        return img_list, n_imgs

    def read_cameras_by_mode(self, mode):
        """Read in all the camera file and keep the index of split"""
        # read cam on all split
        all_mode = ['train', 'eval']
        idx = [[-1]]
        pose_files = []
        intrinsics_files = []
        for i, m in enumerate(all_mode):
            last_idx = idx[i][-1] + 1
            # pose
            pose_dir = osp.join(self.data_spec_dir, self.convert_mode(m), 'pose')
            pose_file = sorted(glob.glob(pose_dir + '/*.txt'))
            pose_files.append(pose_file)

            # intrinsic
            intrinsics_dir = osp.join(self.data_spec_dir, self.convert_mode(m), 'intrinsics')
            intrinsics_file = sorted(glob.glob(intrinsics_dir + '/*.txt'))
            intrinsics_files.append(intrinsics_file)
            idx.append(list(range(last_idx, last_idx + len(pose_file))))

        # train for first, other for last
        split_idx = idx[1] if mode == 'train' else idx[2]

        # concat all the cameras
        cameras = []
        for i, m in enumerate(all_mode):
            for pose_txt, intrinsic_txt in zip(pose_files[i], intrinsics_files[i]):
                assert pose_txt.split('/')[-1] == intrinsic_txt.split('/')[-1]
                cameras.append(self.read_cameras_from_txt(pose_txt, intrinsic_txt))

        return cameras, split_idx

    def read_cameras_from_txt(self, pose_txt, intrinsic_txt):
        """Read camera from txt files """
        with open(pose_txt, 'r') as f:
            pose_lines = f.readline()
        with open(intrinsic_txt, 'r') as f:
            intrinsics_lines = f.readline()

        c2w = self.read_c2w(pose_lines)
        intrinsic = self.read_intrinsic(intrinsics_lines)

        return PerspectiveCamera(intrinsic=intrinsic, c2w=c2w, W=self.W, H=self.H)

    @staticmethod
    def read_c2w(lines):
        """Read c2w from pose file"""
        c2w = np.array([float(x) for x in lines.split(' ')]).reshape(4, 4)

        return c2w

    @staticmethod
    def read_intrinsic(lines):
        """Get intrinsic (3, 3) from line"""
        intrinsic = np.array([float(x) for x in lines.split(' ')]).reshape(4, 4)[:3, :3]

        return intrinsic
