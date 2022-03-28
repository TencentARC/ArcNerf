#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import os.path as osp

import numpy as np

from arcnerf.datasets import get_dataset
from arcnerf.datasets.transform.augmentation import get_transforms
from arcnerf.geometry.transformation import invert_pose
from arcnerf.visual.vis_camera import draw_camera_extrinsic
from tests import setup_test_config

MODE = 'train'
RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)

if __name__ == '__main__':
    cfgs = setup_test_config()
    transforms, _ = get_transforms(getattr(cfgs.dataset, MODE))
    dataset = get_dataset(cfgs.dataset, cfgs.dir.data_dir, None, MODE, transforms)

    spec_result_dir = osp.join(RESULT_DIR, getattr(cfgs.dataset, MODE).type)
    os.makedirs(spec_result_dir, exist_ok=True)

    # get camera pose
    extrinsics = []
    for sample in dataset:
        extrinsics.append(invert_pose(sample['c2w'][None, ...]))
    extrinsics = np.concatenate(extrinsics, axis=0)

    # draw extrinsic
    cam_path = '{}/vis_camera.png'.format(spec_result_dir)
    draw_camera_extrinsic(extrinsics, save_path=cam_path)
