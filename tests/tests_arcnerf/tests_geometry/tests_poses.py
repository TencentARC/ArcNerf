#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
import torch

from arcnerf.geometry.poses import invert_pose
from tests.tests_arcnerf.tests_geometry import TestGeomDict


class TestDict(TestGeomDict):

    def setUp(self):
        super().setUp()

    def tests_invert_pose(self):
        c2w_new = invert_pose(invert_pose(self.c2w))
        self.assertTrue(torch.allclose(self.c2w, c2w_new, atol=1e-3))
        c2w_np = self.c2w.numpy()
        c2w_np_new = invert_pose(invert_pose(c2w_np))
        self.assertTrue(np.allclose(c2w_np, c2w_np_new, atol=1e-3))

    @staticmethod
    def get_max_abs_error(a, b):
        return float((a - b).abs().max())


if __name__ == '__main__':
    unittest.main()
