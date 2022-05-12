# -*- coding: utf-8 -*-

import importlib
import os.path as osp
from copy import deepcopy

from .full_model import FullModel
from common.utils.cfgs_utils import dict_to_obj, obj_to_dict, valid_key_in_cfgs
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
        logger: logger for logging
    """
    cfgs = deepcopy(cfgs)

    # get full model by fg and bkg
    fg_model = MODEL_REGISTRY.get(cfgs.model.type)(cfgs)
    bkg_model_cfgs, bkg_model = None, None
    if valid_key_in_cfgs(cfgs.model, 'background'):
        bkg_model_cfgs = dict_to_obj({'model': obj_to_dict(cfgs.model.background)})
        bkg_model = MODEL_REGISTRY.get(bkg_model_cfgs.model.type)(bkg_model_cfgs)
    model = FullModel(cfgs, fg_model, bkg_model_cfgs, bkg_model)

    if logger is not None:
        logger.add_log('Model type : {}'.format(cfgs.model.type))
        logger.add_log('Model Configs: {}'.format(obj_to_dict(cfgs.model)))

    return model
