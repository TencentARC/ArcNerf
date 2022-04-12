#!/usr/bin/python
# -*- coding: utf-8 -*-

from itertools import combinations
import unittest

import torch
import torch.nn as nn

from arcnerf.models.base_modules import (
    Embedder,
    DenseLayer,
    get_activation,
    GeoNet,
    RadianceNet,
    Sine,
    SirenLayer,
)
from common.utils.cfgs_utils import dict_to_obj


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

    def tests_dense_layer(self):
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

    def tests_get_activation(self):
        types = ['relu', 'softplus', 'leakyrelu', 'sine', 'sigmoid']
        act_types = [nn.ReLU, nn.Softplus, nn.LeakyReLU, Sine, nn.Sigmoid]
        for i, act_type in enumerate(types):
            cfg = {'type': act_type, 'slope': 1e-2, 'w': 30, 'beta': 100}
            cfg = dict_to_obj(cfg)
            act = get_activation(cfg)
            self.assertIsInstance(act, act_types[i])

    def tests_geonet(self):
        x = torch.ones((self.batch_size, 3))
        # normal case
        model = GeoNet(input_ch=3)
        y, feat = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        self.assertEqual(feat.shape, (self.batch_size, 256))
        # W_feat <= 0
        model = GeoNet(input_ch=3, W_feat=0)
        y, _ = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        # multi skips
        model = GeoNet(input_ch=3, skips=[1, 2], skip_reduce_output=True)
        y, feat = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        # act
        cfg = {'type': 'softplus', 'beta': 100}
        cfg = dict_to_obj(cfg)
        model = GeoNet(input_ch=3, act_cfg=cfg)
        y, feat = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        self.assertEqual(feat.shape, (self.batch_size, 256))
        # siren
        model = GeoNet(input_ch=3, use_siren=True, skips=[])
        y, feat = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        self.assertEqual(feat.shape, (self.batch_size, 256))
        # forward with normal output and geo value only
        model = GeoNet(input_ch=3)
        geo_value, feat, grad = model.forward_with_grad(x)
        self.assertEqual(x.shape, grad.shape)
        self.assertEqual(feat.shape, (self.batch_size, 256))
        geo_value = model.forward_geo_value(x)
        self.assertEqual(geo_value.shape, (self.batch_size, 1))

    def tests_radiancenet(self):
        xyz = torch.ones((self.batch_size, 3))
        view_dirs = torch.ones((self.batch_size, 3))
        normals = torch.ones((self.batch_size, 3))
        feat = torch.ones((self.batch_size, 256))
        modes = ['p', 'v', 'n', 'f']
        modes = sum([list(map(list, combinations(modes, i))) for i in range(len(modes) + 1)], [])
        for mode in modes:
            if len(mode) == 0:
                continue
            mode = ''.join(mode)
            model = RadianceNet(mode=mode, W=128, D=8, W_feat_in=256)
            y = model(xyz, view_dirs, normals, feat)
            self.assertEqual(y.shape, (self.batch_size, 3))


if __name__ == '__main__':
    unittest.main()
