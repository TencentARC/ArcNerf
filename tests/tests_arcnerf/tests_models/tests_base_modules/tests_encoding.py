#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.models.base_modules.encoding import FreqEmbedder
from arcnerf.models.base_modules.encoding.gaussian_encoder import GaussianEmbedder


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10

    def tests_freq_embedder(self):
        input_dims = range(1, 10)
        n_freqs = [0, 5, 10]
        periodic_fns = (torch.sin, torch.cos)
        include_inputs = [True, False]
        for input_dim in input_dims:
            for freq in n_freqs:
                for include in include_inputs:
                    if freq == 0 and include is False:
                        continue  # this case is not allowed
                    xyz = torch.ones((self.batch_size, input_dim))
                    model = FreqEmbedder(input_dim, freq, include_input=include, periodic_fns=periodic_fns)
                    out = model(xyz)
                    out_dim = input_dim * (len(periodic_fns) * freq + include)
                    self.assertEqual(out.shape, (self.batch_size, out_dim))

    def test_gaussian_embedder(self):
        n_interval = 20
        ipe_embed_freqs = [1, 5, 10]
        gaussian_fns = ['cone', 'cylinder']
        near, far = 0.0, 2.0
        # prepare inputs
        zvals = torch.linspace(near, far, n_interval + 1).unsqueeze(0)
        zvals = torch.repeat_interleave(zvals, self.batch_size, 0)  # (B, N+1)
        rays_o = torch.rand((self.batch_size, 3))
        rays_d = torch.rand((self.batch_size, 3))
        rays_i = torch.rand((self.batch_size, 1))
        for gaussian_fn in gaussian_fns:
            for ipe_embed_freq in ipe_embed_freqs:
                model = GaussianEmbedder(gaussian_fn, ipe_embed_freq)
                out = model(zvals, rays_o, rays_d, rays_i)
                self.assertEqual(out.shape, (self.batch_size, n_interval, 6 * ipe_embed_freq))
