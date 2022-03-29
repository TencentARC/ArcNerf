#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
import torch

from arcnerf.geometry.projection import pixel_to_cam
from arcnerf.geometry.transformation import normalize
from tests.tests_arcnerf.tests_geometry import TestGeomDict


class TestDict(TestGeomDict):

    def setUp(self):
        super().setUp()

    def tests_normalization(self):
        vec = pixel_to_cam(self.pixels, self.depth, self.intrinsic)
        norm_vec = normalize(vec)
        self.assertTrue(torch.allclose(torch.ones(size=norm_vec.shape[:2]), torch.norm(norm_vec, dim=-1)))
        vec_np = vec.numpy()
        norm_vec_np = normalize(vec_np)
        self.assertTrue(np.allclose(np.ones(shape=norm_vec_np.shape[:2]), np.linalg.norm(norm_vec_np, axis=-1)))

    @staticmethod
    def get_max_abs_error(a, b):
        return float((a - b).abs().max())


if __name__ == '__main__':
    unittest.main()
