#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.models.base_modules.encoding import FreqEmbedder, GaussianEmbedder, Gaussian, SHEmbedder


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

    def tests_sh_embedder(self):
        # test freq factors, at most 5
        include_inputs = [True, False]
        for degree in range(1, 6):
            for include in include_inputs:
                model = SHEmbedder(n_freqs=degree, include_input=include)
                xyz = torch.ones((self.batch_size, 3))
                out = model(xyz)
                out_dim = model.get_output_dim()
                self.assertEqual(out_dim, degree**2 + include * 3)
                self.assertEqual(out.shape, (self.batch_size, degree**2 + include * 3))

    def test_gaussian_embedder(self):
        n_interval = 20
        n_freqs = [0, 5, 10]
        include_inputs = [True, False]
        gaussian_fns = ['cone', 'cylinder']
        near, far = 0.0, 2.0
        # prepare inputs
        zvals = torch.linspace(near, far, n_interval + 1).unsqueeze(0)
        zvals = torch.repeat_interleave(zvals, self.batch_size, 0)  # (B, N+1)
        rays_o = torch.rand((self.batch_size, 3))
        rays_d = torch.rand((self.batch_size, 3))
        rays_r = torch.rand((self.batch_size, 1))
        for gaussian_fn in gaussian_fns:
            for freq in n_freqs:
                for include in include_inputs:
                    if freq == 0 and include is False:
                        continue  # this case is not allowed
                    gaussian = Gaussian(gaussian_fn=gaussian_fn)
                    mean_cov = gaussian(zvals, rays_o, rays_d, rays_r)
                    mean_cov = mean_cov.view(-1, 6)  # (BN, 6)
                    model = GaussianEmbedder(3, freq, include_input=include)
                    out = model(mean_cov)
                    out_dim = 3 * (2 * freq + include)
                    self.assertEqual(out.shape, (self.batch_size * n_interval, out_dim))
