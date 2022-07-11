# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.ops import SHEncode


class TestModelDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 4096
        cls.n_rays = 72 * 35

    def tests_sh_encode(self):
        if not torch.cuda.is_available():
            return

        for degree in range(1, 4):
            func = SHEncode(degree)
            xyz = torch.rand((4096, 3), dtype=torch.float32, requires_grad=True).cuda()
            out = func(xyz)
            self.assertEqual(out.shape, (self.batch_size, degree**2))

            loss = torch.sum(1 - out**2)
            loss.backward()
