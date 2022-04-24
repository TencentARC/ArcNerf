#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

import torch

from common.metric.metric_dict import MetricDictCounter
from common.utils.cfgs_utils import parse_configs, get_value_from_cfgs_field
from common.utils.logger import Logger
from common.utils.model_io import load_model
from arcnerf.datasets import get_dataset, get_model_feed_in
from arcnerf.datasets.transform.augmentation import get_transforms
from arcnerf.eval.eval_func import run_eval
from arcnerf.metric import build_metric
from arcnerf.models import build_model
from arcnerf.visual.render_img import render_progress_imgs, write_progress_imgs

if __name__ == '__main__':
    cfgs = parse_configs()
    cfgs.dist.rank = 0
    cfgs.dist.local_rank = 0
    cfgs.dist.world_size = 1

    # device. Use gpu only when available and specified
    assert isinstance(cfgs.gpu_ids, int), 'Please use cpu or a single gpu for eval... (-1: cpu; 0: cuda0)'
    device = 'cpu'
    if torch.cuda.is_available() and cfgs.gpu_ids >= 0:
        device = 'gpu'
        torch.cuda.set_device(cfgs.gpu_ids)

    # set eval_dir
    assert cfgs.dir.eval_dir is not None, 'Please specify the eval_dir for saving results...'
    eval_dir = cfgs.dir.eval_dir
    os.makedirs(eval_dir, exist_ok=True)

    # set logger
    logger = Logger(rank=0)
    logger.add_log('Eval on test data... Result write to {}...'.format(cfgs.dir.eval_dir))

    # set dataset
    eval_bs = get_value_from_cfgs_field(cfgs.dataset.eval, 'eval_batch_size', 1)
    tkwargs_eval = {'batch_size': eval_bs, 'num_workers': cfgs.worker, 'pin_memory': True, 'drop_last': False}
    eval_transform, _ = get_transforms(getattr(cfgs.dataset, 'eval'))
    dataset = get_dataset(cfgs.dataset, cfgs.dir.data_dir, logger=logger, mode='eval', transfroms=eval_transform)

    loader = torch.utils.data.DataLoader(dataset, **tkwargs_eval)

    # set and load model
    assert cfgs.model_pt is not None, 'Please specify the model_pt for evaluation...'
    model = build_model(cfgs, logger)
    model = load_model(logger, model, None, cfgs.model_pt, cfgs)

    if device == 'gpu':
        model.cuda()

    # set metric
    eval_metric = build_metric(cfgs, logger)
    metric_dict = MetricDictCounter()

    # eval
    metric_info, files = run_eval(
        loader, get_model_feed_in, model, logger, eval_metric, metric_dict, device, render_progress_imgs,
        cfgs.progress.max_samples_eval
    )

    # write down results
    if metric_info is None:
        logger.add_log('No evaluation perform...', level='warning')
        exit()

    if files is not None and len(files) > 0:
        write_progress_imgs(files, eval_dir, eval=True)
        logger.add_log('Visual results add to {}'.format(eval_dir))

    logger.add_log('Evaluation Benchmark result. \n {}'.format(metric_info))

    # write to log file as well
    eval_log_file = os.path.join(cfgs.dir.eval_dir, 'eval_log.txt')
    with open(eval_log_file, 'w') as f:
        f.writelines('Model path {}\n'.format(cfgs.model_pt))
        f.writelines(metric_info)
