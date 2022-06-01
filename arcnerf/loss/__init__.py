# -*- coding: utf-8 -*-

import importlib
import os.path as osp
from copy import deepcopy

from common.utils.cfgs_utils import obj_to_dict
from common.utils.file_utils import scan_dir
from common.utils.registry import LOSS_REGISTRY

__all__ = ['build_loss']

loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in scan_dir(loss_folder) if v.endswith('_loss.py')]
_loss_modules = [importlib.import_module(f'arcnerf.loss.{file_name}') for file_name in loss_filenames]


def build_loss(cfgs, logger):
    """Build loss factory from configs.

    Args:
        cfgs (dict): Configuration.
        logger: logger for logging
    """
    cfgs = deepcopy(cfgs)
    loss_names = []
    loss_weights = []
    loss_funcs = []
    for loss in cfgs.loss.__dict__:
        loss_funcs.append(LOSS_REGISTRY.get(loss)(getattr(cfgs.loss, loss)))
        loss_names.append(loss)
        loss_weights.append(getattr(cfgs.loss, loss).weight)
    loss_factory = AllLoss(loss_funcs, loss_names, loss_weights)
    if logger is not None:
        logger.add_log('Loss types : {}'.format(loss_names))
        logger.add_log('Loss Weights: {}'.format(loss_weights))
        logger.add_log('Loss dict: {}'.format(obj_to_dict(cfgs.loss)))

    return loss_factory


class AllLoss(object):
    """All loss combine. Weights will be multiplied here.
        For all the loss, you should change var from inputs to output's device for calculation
    """

    def __init__(self, loss_funcs, loss_names, loss_weights):
        super(AllLoss).__init__()
        self.loss_funcs = loss_funcs
        self.loss_names = loss_names
        self.loss_weights = loss_weights
        assert len(self.loss_funcs) == len(self.loss_weights), 'Num of loss and weight not matched...'
        self.num_loss = len(loss_funcs)

    def __call__(self, inputs, output):
        loss = {}
        loss['sum'] = 0.0
        loss['names'] = []
        for i, l in enumerate(self.loss_funcs):
            loss[self.loss_names[i]] = l(inputs, output) * self.loss_weights[i]
            loss['sum'] += loss[self.loss_names[i]]
            loss['names'].append(self.loss_names[i])

        return loss
