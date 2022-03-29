#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.geometry.poses import invert_pose
from tests import setup_test_config


class TestGeomDict(unittest.TestCase):

    def setUp(self):
        self.cfgs = setup_test_config()
        self.H, self.W = 480, 640
        self.batch_size = 2
        self.intrinsic, self.c2w = self.setup_params()
        self.pixels, self.depth = self.set_pixels()

    def setup_params(self):
        intrinsic = torch.eye(3, dtype=torch.float32)
        intrinsic[0, 2] = self.W / 2.0
        intrinsic[1, 2] = self.H / 2.0
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, :3] = torch.rand(size=(3, 3))
        intrinsic = torch.repeat_interleave(intrinsic[None, ...], self.batch_size, dim=0)
        c2w = torch.repeat_interleave(c2w[None, ...], self.batch_size, dim=0)

        return intrinsic, c2w

    def set_pixels(self):
        i, j = torch.meshgrid(
            torch.linspace(0, self.W - 1, self.W), torch.linspace(0, self.H - 1, self.H)
        )  # i, j: (W, H)
        pixels = torch.repeat_interleave(
            torch.stack([i, j]).reshape(-1, 2).unsqueeze(0), self.batch_size, dim=0
        )  # (2, WH, 2)
        depth = torch.ones(size=(self.batch_size, pixels.shape[1]))

        return pixels, depth

    def get_w2c(self):
        w2c = invert_pose(self.c2w)

        return w2c
