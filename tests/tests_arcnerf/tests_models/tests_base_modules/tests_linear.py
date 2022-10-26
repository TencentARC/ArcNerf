#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from arcnerf.models.base_modules import DenseLayer, SirenLayer


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10

    def tests_dense_layer(self):
        """Test dense/siren layers for different in/out size"""
        input_dims = range(1, 10)
        out_dims = range(1, 10)
        for i_dim in input_dims:
            for o_dim in out_dims:
                model = DenseLayer(i_dim, o_dim)
                model_s = SirenLayer(i_dim, o_dim)
                x = torch.ones((self.batch_size, i_dim))
                y = model(x)
                y_s = model_s(x)
                self.assertEqual(y.shape, (self.batch_size, o_dim))
                self.assertEqual(y_s.shape, (self.batch_size, o_dim))
