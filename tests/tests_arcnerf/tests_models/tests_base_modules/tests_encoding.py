#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import torch

from arcnerf.geometry.transformation import normalize
from arcnerf.models.base_modules.encoding import (
    build_encoder, DenseGridEmbedder, FreqEmbedder, GaussianEmbedder, Gaussian, HashGridEmbedder, SHEmbedder
)
from common.utils.cfgs_utils import dict_to_obj
from common.utils.logger import Logger
from tests.tests_arcnerf.tests_ops import log_custom_benchmark

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 4096
        cls.logger = Logger(path=osp.join(RESULT_DIR, './encoder.txt'), keep_console=False)

    def check_output_and_grad(self, out_torch, out_custom, out_custom_forward_only, grad_torch, grad_custom, atol=1e-8):
        """Check the output and grad"""
        if out_torch is not None:
            if isinstance(out_torch, list):
                for out, _out, _out_forward in zip(out_torch, out_custom, out_custom_forward_only):
                    if isinstance(out, torch.Tensor):
                        self.assertTrue(torch.allclose(out, _out, atol=atol))
                        self.assertTrue(torch.allclose(out, _out_forward, atol=atol))
            else:
                if isinstance(out_torch, torch.Tensor):
                    self.assertTrue(torch.allclose(out_torch, out_custom, atol=atol))
                    self.assertTrue(torch.allclose(out_torch, out_custom_forward_only, atol=atol))

        if grad_torch is not None:
            if isinstance(grad_torch, list):
                for grad, _grad in zip(grad_torch, grad_custom):
                    if isinstance(grad, torch.Tensor):
                        self.assertTrue(torch.allclose(grad, _grad, atol=atol))
            else:
                if isinstance(grad_torch, torch.Tensor):
                    self.assertTrue(torch.allclose(grad_torch, grad_custom, atol=atol))

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
                    xyz = torch.rand((self.batch_size, input_dim))
                    model = FreqEmbedder(input_dim, freq, include_input=include, periodic_fns=periodic_fns)
                    out = model(xyz)
                    out_dim = input_dim * (len(periodic_fns) * freq + include)
                    self.assertEqual(out.shape, (self.batch_size, out_dim))

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

    def tests_densegrid_encoder(self):
        for W_feat in [0, 256]:
            model = DenseGridEmbedder(include_input=True, W_feat=W_feat, side=2.0).cuda()
            xyz = torch.rand((self.batch_size, 3)).cuda()
            out = model(xyz)
            out_dim = model.get_output_dim()
            self.assertEqual(out_dim, 1 + 3 + W_feat)
            self.assertEqual(out.shape, (self.batch_size, 1 + 3 + W_feat))

    def tests_sh_embedder(self):
        # test freq factors, at most 5
        include_inputs = [True, False]
        for degree in range(1, 6):
            for include in include_inputs:
                model = SHEmbedder(n_freqs=degree, include_input=include)
                xyz = torch.rand((self.batch_size, 3))
                out = model(xyz)
                out_dim = model.get_output_dim()
                self.assertEqual(out_dim, degree**2 + include * 3)
                self.assertEqual(out.shape, (self.batch_size, degree**2 + include * 3))

    def tests_sh_embedder_comp(self):
        if not torch.cuda.is_available():
            return

        # double gets the check
        dirs = torch.rand((self.batch_size, 3), dtype=torch.double)
        dirs_norm = dirs / normalize(dirs)

        for degree in range(1, 6):
            sh_torch = SHEmbedder(n_freqs=degree, include_input=True)
            sh_custom = SHEmbedder(n_freqs=degree, include_input=True, backend='cuda')

            inputs = [dirs_norm.clone().detach().requires_grad_(True)]

            out_torch, out_custom, out_custom_forward_only, grad_torch, grad_custom = log_custom_benchmark(
                self.logger, 'SH Encode(degree {})'.format(degree), sh_torch, sh_custom, inputs
            )

            # the accumulate grad gets quite large error
            self.check_output_and_grad(out_torch, out_custom, out_custom_forward_only, grad_torch, grad_custom)

    def tests_hashgrid_encoder(self):
        n_levels = 16
        n_feat_per_entry = 2
        side = 1.5  # to make pts outside the volume
        model = HashGridEmbedder(n_levels=n_levels, n_feat_per_entry=n_feat_per_entry, side=side, include_input=True)
        xyz = torch.rand((self.batch_size, 3))
        out = model(xyz)
        out_dim = model.get_output_dim()
        self.assertEqual(out_dim, n_levels * n_feat_per_entry + 3)
        self.assertEqual(out.shape, (self.batch_size, n_levels * n_feat_per_entry + 3))

    def tests_hashgrid_embedder_comp(self):
        if not torch.cuda.is_available():
            return

        n_levels = [8, 16]
        n_feat_per_entry = [2, 4, 8]
        hashmap_size = 19
        side = 1.5  # to make pts outside the volume

        # double gets the check
        xyz = torch.rand((self.batch_size, 3), dtype=torch.double)

        for level in n_levels:
            for n_feat in n_feat_per_entry:
                hashgrid_torch = HashGridEmbedder(
                    n_levels=level, n_feat_per_entry=n_feat, hashmap_size=hashmap_size, side=side, include_input=True
                ).double().cuda()  # embeddings param needs double
                embeddings_data = hashgrid_torch.get_embeddings()
                hashgrid_custom = HashGridEmbedder(
                    n_levels=level,
                    n_feat_per_entry=n_feat,
                    hashmap_size=hashmap_size,
                    side=side,
                    include_input=True,
                    backend='cuda'
                ).double().cuda()  # embeddings param needs double
                hashgrid_custom.set_embeddings(embeddings_data.clone())  # make sure use the same embeddings

                inputs = [xyz.clone().detach().requires_grad_(True)]

                out_torch, out_custom, out_custom_forward_only, grad_torch, grad_custom = log_custom_benchmark(
                    self.logger,
                    'HashGrid Encode(n_level {} - n_feat {})'.format(level, n_feat),
                    hashgrid_torch,
                    hashgrid_custom,
                    inputs,
                    n_iter=1
                )

                # the accumulate grad gets quite large error
                self.check_output_and_grad(
                    out_torch, out_custom, out_custom_forward_only, grad_torch, grad_custom, atol=1e-4
                )  # 5e-5 level

    def tests_composite_encoder(self):
        # that is the feat used in nsvf
        composite_cfgs = {
            'type': 'CompositeEmbedder',
            'sub_encoder_types': ['DenseGridEmbedder', 'FreqEmbedder'],
            'input_dim': 3,
            'n_freqs': 0,
            'sub_encoder1': {
                'type': 'DenseGridEmbedder',
                'include_input': False,
                'feat_only': True,
                'n_grid': 128,
                'side': 1,
                'W_feat': 32
            },
            'sub_encoder2': {
                'type': 'FreqEmbedder',
                'n_freqs': 6
            }
        }

        model, _, _ = build_encoder(dict_to_obj(composite_cfgs))
        xyz = torch.rand((self.batch_size, 3))
        out = model(xyz)
        out_dim = model.get_output_dim()
        self.assertEqual(out_dim, out_dim)
        self.assertEqual(out.shape, (self.batch_size, out_dim))
