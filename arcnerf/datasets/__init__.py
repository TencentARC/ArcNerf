# -*- coding: utf-8 -*-

import importlib
import os.path as osp
from copy import deepcopy

from common.utils.cfgs_utils import obj_to_dict
from common.utils.file_utils import scan_dir
from common.utils.registry import DATASET_REGISTRY

__all__ = ['get_dataset']

datasets_folder = osp.dirname(osp.abspath(__file__))
datasets_filenames = [osp.splitext(osp.basename(v))[0] for v in scan_dir(datasets_folder) if v.endswith('_dataset.py')]
_dataset_modules = [importlib.import_module(f'arcnerf.datasets.{file_name}') for file_name in datasets_filenames]


def get_mode_cfgs(cfgs, mode='train'):
    return getattr(cfgs, mode, 'train')


def get_dataset(cfgs, data_dir, logger, mode='train', transfroms=None):
    """Build dataset from configs.

    Args:
        cfgs (dict): Configuration.
        mode: control workflow for different split
        data_dir: main data_dir storing the data
    """
    cfgs = deepcopy(cfgs)
    cfgs_mode = get_mode_cfgs(cfgs, mode)
    model = DATASET_REGISTRY.get(getattr(cfgs_mode, 'type'))(cfgs_mode, data_dir, mode, transfroms)
    if logger is not None:
        logger.add_log('Dataset type : {} - mode: {}'.format(getattr(cfgs_mode, 'type'), mode))
        logger.add_log('Dataset Configs: {}'.format(obj_to_dict(cfgs_mode)))

    return model
