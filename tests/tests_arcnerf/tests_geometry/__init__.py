#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import torch

from arcnerf.geometry.poses import invert_poses
from tests import setup_test_config

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestGeomDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfgs = setup_test_config()
        cls.H, cls.W = 480, 640
        cls.focal = 1000.0
        cls.skewness = 10.0
        cls.batch_size = 2
        cls.intrinsic, cls.c2w = cls.setup_params()
        cls.pixels, cls.depth = cls.set_pixels()

    @classmethod
    def setup_params(cls):
        intrinsic = torch.eye(3, dtype=torch.float32)
        intrinsic[0, 0] = cls.focal
        intrinsic[1, 1] = cls.focal
        intrinsic[0, 1] = cls.skewness
        intrinsic[0, 2] = cls.W / 2.0
        intrinsic[1, 2] = cls.H / 2.0
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, :4] = torch.rand(size=(3, 4))
        intrinsic = torch.repeat_interleave(intrinsic[None, ...], cls.batch_size, dim=0)
        c2w = torch.repeat_interleave(c2w[None, ...], cls.batch_size, dim=0)

        return intrinsic, c2w

    @classmethod
    def set_pixels(cls):
        i, j = torch.meshgrid(torch.linspace(0, cls.W - 1, cls.W), torch.linspace(0, cls.H - 1, cls.H))  # i, j: (W, H)
        pixels = torch.repeat_interleave(
            torch.stack([i, j]).reshape(-1, 2).unsqueeze(0), cls.batch_size, dim=0
        )  # (2, WH, 2)
        depth = torch.ones(size=(cls.batch_size, pixels.shape[1]))

        return pixels, depth

    def get_w2c(self):
        w2c = invert_poses(self.c2w)

        return w2c
