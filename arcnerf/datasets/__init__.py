# -*- coding: utf-8 -*-

import importlib
import os.path as osp
from copy import deepcopy

from common.utils.cfgs_utils import obj_to_dict
from common.utils.file_utils import scan_dir
from common.utils.registry import DATASET_REGISTRY

__all__ = ['get_dataset', 'get_model_feed_in', 'POTENTIAL_KEYS']

datasets_folder = osp.dirname(osp.abspath(__file__))
datasets_filenames = [osp.splitext(osp.basename(v))[0] for v in scan_dir(datasets_folder) if v.endswith('_dataset.py')]
_dataset_modules = [importlib.import_module(f'arcnerf.datasets.{file_name}') for file_name in datasets_filenames]

POTENTIAL_KEYS = ['img', 'mask', 'rays_o', 'rays_d', 'rays_r', 'bounds', 'bkg_color', 'exp_time']


def get_mode_cfgs(cfgs, mode='train'):
    return getattr(cfgs, mode, 'train')


def get_dataset(cfgs, data_dir, logger, mode='train', transfroms=None):
    """Build dataset from configs.

    Args:
        cfgs (dict): Configuration.
        data_dir: main data_dir storing the data
        logger: logger for logging
        mode: control workflow for different split
        transfroms: the transforms/augmentation used for dataset
    """
    cfgs = deepcopy(cfgs)
    cfgs_mode = get_mode_cfgs(cfgs, mode)
    dataset = DATASET_REGISTRY.get(getattr(cfgs_mode, 'type'))(cfgs_mode, data_dir, mode, transfroms)
    if logger is not None:
        logger.add_log('Dataset type : {} - mode: {}'.format(getattr(cfgs_mode, 'type'), mode))
        logger.add_log('Dataset Configs: {}'.format(obj_to_dict(cfgs_mode)))
        logger.add_log('Dataset Length: {}'.format(len(dataset)))

    return dataset


def get_model_feed_in(inputs, device):
    """Get the core model feed in and put it to the model's device
    device is only 'cpu' or 'gpu'
    """
    feed_in = {}
    for key in POTENTIAL_KEYS:
        if key in inputs:
            feed_in[key] = inputs[key]
            if device == 'gpu':
                feed_in[key] = feed_in[key].cuda(non_blocking=True)

    # rays must be there
    batch_size = inputs['rays_o'].shape[0]

    return feed_in, batch_size
