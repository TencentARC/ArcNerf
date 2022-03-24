# -*- coding: utf-8 -*-

import importlib
import os.path as osp
from copy import deepcopy

from common.utils.cfgs_utils import obj_to_dict
from common.utils.file_utils import scan_dir
from common.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scan_dir(model_folder) if v.endswith('_model.py')]
_model_modules = [importlib.import_module(f'arcnerf.models.{file_name}') for file_name in model_filenames]


def build_model(cfgs, logger):
    """Build model from configs.

    Args:
        cfgs (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    cfgs = deepcopy(cfgs)
    model = MODEL_REGISTRY.get(cfgs.model.type)(cfgs)
    if logger is not None:
        logger.add_log('Model type : {}'.format(cfgs.model.type))
        logger.add_log('Model Configs: {}'.format(obj_to_dict(cfgs.model)))

    return model
