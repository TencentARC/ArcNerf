# -*- coding: utf-8 -*-

import os.path as osp

from common.utils.cfgs_utils import load_configs

CONFIG = osp.abspath(osp.join(__file__, '../../../..', 'configs', 'datasets', 'DTU', '65.yaml'))


def setup_test_config(unknowns=None):
    """Set up config"""
    cfgs = load_configs(CONFIG, unknowns)

    return cfgs
