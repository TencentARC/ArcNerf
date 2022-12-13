# -*- coding: utf-8 -*-

import importlib
import os.path as osp
from copy import deepcopy

from common.utils.file_utils import scan_dir
from common.utils.registry import METRIC_REGISTRY

__all__ = ['build_metric']

metric_folder = osp.dirname(osp.abspath(__file__))
metric_filenames = [osp.splitext(osp.basename(v))[0] for v in scan_dir(metric_folder) if v.endswith('_metric.py')]
_metric_modules = [importlib.import_module(f'simplenerf.metric.{file_name}') for file_name in metric_filenames]


def build_metric(cfgs, logger):
    """Build metric calculator from configs.

    Args:
        cfgs (dict): Configuration.
        logger: logger for logging
    """
    cfgs = deepcopy(cfgs)
    metric_names = []
    metric_funcs = []
    for metric in cfgs.metric.__dict__:
        metric_funcs.append(METRIC_REGISTRY.get(metric)(getattr(cfgs.metric, metric)))
        metric_names.append(metric)
    metric_factory = AllMetric(metric_funcs, metric_names)
    if logger is not None:
        logger.add_log('Metric types : {}'.format(metric_names))

    return metric_factory


class AllMetric(object):
    """All Metric separately.
    For all the metric, you should change var from inputs to output's device for calculation
    """

    def __init__(self, metric_funcs, metric_names):
        super(AllMetric).__init__()
        self.metric_funcs = metric_funcs
        self.metric_names = metric_names
        self.num_metric = len(metric_funcs)

    def __call__(self, inputs, output):
        metric = {}
        metric['names'] = []
        for i, m in enumerate(self.metric_funcs):
            metric[self.metric_names[i]] = m(inputs, output)
            metric['names'].append(self.metric_names[i])

        return metric
