#!/usr/bin/python
# -*- coding: utf-8 -*-
# run `python tests/test_utils/tests_cfgs.py --configs ./configs/default.yaml --sda.sdse sdas \
#                                            --debug.debug_mode True --a.b.c.d '['111', 123]'` for checking

import unittest

from common.utils.cfgs_utils import obj_to_dict
from tests import setup_test_config


class TestDict(unittest.TestCase):

    def setUp(self):
        self.cfgs = setup_test_config()

    def test_cfg(self):
        self.assertIsNotNone(self.cfgs)

    def test_obj_to_dict(self):
        self.assertIsInstance(obj_to_dict(self.cfgs), dict)

    def test_unknowns_remap(self):
        """Test remapping values from unknowns"""
        unknowns = [
            '--str', 'str_test', '--true', 'True', '--false', 'False', '--none', 'None', '--list', '0,1,2',
            '--list_bracket', '[3, 2, 3]', '--list_str', '[s1, s2, s3]', '--int_p', '2', '--int_n', '-2', '--float_p',
            '3.12', '--float_n', '-3.12', '--sci_p', '1e4', '--sci_n', '-1e4', '--sci_p_small', '1e-4', '--sci_n_small',
            '-1e-4'
        ]
        self.cfgs = setup_test_config(unknowns)
        self.assertEqual(self.cfgs.str, 'str_test')
        self.assertTrue(self.cfgs.true)
        self.assertFalse(self.cfgs.false)
        self.assertIsNone(self.cfgs.none)
        self.assertListEqual(self.cfgs.list, [0, 1, 2])
        self.assertListEqual(self.cfgs.list_bracket, [3, 2, 3])
        self.assertListEqual(self.cfgs.list_str, ['s1', 's2', 's3'])
        self.assertEqual(self.cfgs.int_p, 2)
        self.assertEqual(self.cfgs.int_n, -2)
        self.assertAlmostEqual(self.cfgs.float_p, 3.12)
        self.assertAlmostEqual(self.cfgs.float_n, -3.12)
        self.assertAlmostEqual(self.cfgs.sci_p, 1e4)
        self.assertAlmostEqual(self.cfgs.sci_n, -1e4)
        self.assertAlmostEqual(self.cfgs.sci_p_small, 1e-4)
        self.assertAlmostEqual(self.cfgs.sci_n_small, -1e-4)


if __name__ == '__main__':
    unittest.main()
