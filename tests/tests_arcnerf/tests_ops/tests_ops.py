# -*- coding: utf-8 -*-

import math
import unittest

import torch
from torch.autograd import gradcheck

from arcnerf.ops import SHEncode, HashGridEncode


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
            # only double gets the accuracy
            xyz = torch.rand((4096, 3), dtype=torch.double, requires_grad=True).cuda()
            out = func(xyz)
            self.assertEqual(out.shape, (self.batch_size, degree**2))

            loss = torch.sum(1 - out**2)
            loss.backward()

            # auto check
            self.assertTrue(gradcheck(func, xyz, eps=1e-6, atol=1e-8))

    def tests_hashgrid_encode(self):
        if not torch.cuda.is_available():
            return

        # set up inputs
        n_levels = 16
        n_feat_per_entry = 2
        hashmap_size = 2**19
        base_res, max_res = 16, 512
        per_level_scale = math.exp((math.log((max_res / base_res))) / (float(n_levels) - 1))

        offsets = []
        resolutions = []
        n_total_embed = 0
        for i in range(n_levels):
            offsets.append(n_total_embed)
            cur_res = math.floor(base_res * per_level_scale**i)
            resolutions.append(cur_res)
            n_embed_per_level = min(hashmap_size, (cur_res + 1)**3)  # save memory for low res
            n_total_embed += n_embed_per_level
        offsets.append(n_total_embed)

        std = 1e-4
        embeddings = torch.empty(
            n_total_embed, n_feat_per_entry, dtype=torch.double
        ).clone().detach().requires_grad_(True).cuda()
        embeddings.uniform_(-std, std)

        min_xyz = [-0.75, -0.75, -0.75]
        max_xyz = [0.75, 0.75, 0.75]

        func = HashGridEncode(n_levels, n_feat_per_entry, offsets, resolutions).cuda()
        xyz = torch.rand((4096, 3), dtype=torch.double, requires_grad=True).cuda()
        out = func(xyz, embeddings, min_xyz, max_xyz)

        self.assertEqual(out.shape, (self.batch_size, n_levels * n_feat_per_entry))
        exit()

        loss = torch.sum(1 - out**2)
        loss.backward()

        # auto check
        self.assertTrue(gradcheck(func, xyz, eps=1e-6, atol=1e-8))
