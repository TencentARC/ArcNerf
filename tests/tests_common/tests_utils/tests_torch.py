#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
import torch

from common.utils.torch_utils import chunk_processing


class TestDict(unittest.TestCase):

    def setUp(self):
        self.batch_size = 5
        self.n_sample = 16
        self.chunk_size = 10
        self.num_chunk = 8
        self.bn3 = torch.ones(self.batch_size * self.n_sample, 3)
        self.bn = torch.ones(self.batch_size * self.n_sample)
        self.bn3_np = np.ones((self.batch_size * self.n_sample, 3))
        self.bn_np = np.ones((self.batch_size * self.n_sample))
        self.none = None
        self.bool = False
        self.float = 3.0
        self.str = 'test'
        self.other = True
        self.inputs = {
            'bn3': self.bn3,
            'bn': self.bn,
            'bn3_np': self.bn3_np,
            'bn_np': self.bn_np,
            'none': self.none,
            'bool': self.bool,
            'float': self.float,
            'str': self.str,
        }

    def tests_chunk_process(self):
        # list process
        def list_func(bn3, bn, bn3_np, bn_np, none, bool, float, str, other):
            return bn3, bn, bn3_np, bn_np, none, bool, float, str

        bn3_o, bn_o, bn3_np_o, bn_np_o, none_o, bool_o, float_o, str_o = chunk_processing(
            list_func, self.chunk_size, self.bn3, self.bn, self.bn3_np, self.bn_np, self.none, self.bool, self.float,
            self.str, self.other
        )
        self.assertEqual(bn3_o.shape, self.bn3.shape)
        self.assertEqual(bn_o.shape, self.bn.shape)
        self.assertEqual(bn3_np_o.shape, self.bn3_np.shape)
        self.assertEqual(bn_np_o.shape, self.bn_np.shape)
        self.assertTrue(all([v is None for v in none_o]))
        self.assertTrue(all([v is False for v in bool_o]))
        self.assertTrue(all([v == self.float for v in float_o]))
        self.assertTrue(all([v == self.str for v in str_o]))

        # with dict
        def dict_func(input, other):
            return input

        out = chunk_processing(dict_func, self.chunk_size, self.inputs, self.other)
        self.assertEqual(out['bn3'].shape, self.bn3.shape)
        self.assertEqual(out['bn'].shape, self.bn.shape)
        self.assertEqual(out['bn3_np'].shape, self.bn3_np.shape)
        self.assertEqual(out['bn_np'].shape, self.bn_np.shape)
        self.assertTrue(all([v is None for v in out['none']]))
        self.assertTrue(all([v is False for v in out['bool']]))
        self.assertTrue(all([v == self.float for v in out['float']]))
        self.assertTrue(all([v == self.str for v in out['str']]))


if __name__ == '__main__':
    unittest.main()
