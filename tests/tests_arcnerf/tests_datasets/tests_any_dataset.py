#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np

from arcnerf.datasets import get_dataset
from arcnerf.datasets.transform.augmentation import get_transforms
from arcnerf.geometry.transformation import normalize
from arcnerf.visual.plot_3d import draw_3d_components
from tests import setup_test_config

MODE = 'train'
RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    def setUp(self):
        self.cfgs = setup_test_config()
        self.dataset_type = getattr(self.cfgs.dataset, MODE).type
        self.dataset = self.setup_dataset()

    def setup_dataset(self):
        transforms, _ = get_transforms(getattr(self.cfgs.dataset, MODE))
        dataset = get_dataset(self.cfgs.dataset, self.cfgs.dir.data_dir, None, MODE, transforms)

        return dataset

    def tests_get_dataset(self):
        self.assertIsInstance(self.dataset[0], dict)

    def tests_vis_cameras(self):
        c2w = []
        for sample in self.dataset:
            c2w.append(sample['c2w'][None, ...])
        c2w = np.concatenate(c2w, axis=0)  # (n, 4, 4)

        cam_loc = c2w[:, :3, 3]  # (n, 3)
        ori_point = np.array([0.0, 0.0, 0.0], dtype=cam_loc.dtype)[None, :]  # (1, 3)
        rays_d = normalize(ori_point - cam_loc)

        cam_path = '{}/{}_vis_camera.png'.format(RESULT_DIR, self.dataset_type)
        draw_3d_components(
            c2w,
            points=ori_point,
            rays=(cam_loc, rays_d),
            title='{} Cam position'.format(self.dataset_type),
            save_path=cam_path,
        )


if __name__ == '__main__':
    unittest.main()
