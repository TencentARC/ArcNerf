#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.models.base_modules import Embedder


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10

    def tests_embedder(self):
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
                    model = Embedder(input_dim, freq, include_input=include, periodic_fns=periodic_fns)
                    out = model(xyz)
                    self.assertEqual(out.shape[0], self.batch_size)
                    out_dim = input_dim * (len(periodic_fns) * freq + include)
                    self.assertEqual(out.shape[1], out_dim)
