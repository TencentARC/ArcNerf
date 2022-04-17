#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp

import cv2
import torch

from common.metric.metric_dict import MetricDictCounter
from common.utils.cfgs_utils import parse_configs, get_value_from_cfgs_field
from common.utils.logger import Logger
from common.utils.model_io import load_model
from arcnerf.datasets import get_dataset
from arcnerf.datasets.transform.augmentation import get_transforms
from arcnerf.eval.eval_func import run_eval
from arcnerf.metric import build_metric
from arcnerf.models import build_model
from arcnerf.visual.render_img import render_progress_img

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

    # get_model_feed_in_func
    def get_model_feed_in(data, device):
        """Get core model feed in."""
        feed_in = {'img': data['img'], 'mask': data['mask'], 'rays_o': data['rays_o'], 'rays_d': data['rays_d']}

        if device == 'gpu':
            for k in feed_in.keys():
                feed_in[k] = feed_in[k].cuda(non_blocking=True)

        batch_size = data['img'].shape[0]

        return feed_in, batch_size

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
        loader, get_model_feed_in, model, logger, eval_metric, metric_dict, device, render_progress_img,
        cfgs.progress.max_samples_eval
    )

    # write down results
    if metric_info is None:
        logger.add_log('Not evaluation perform...', level='warning')
        exit()

    if files is not None and len(files) > 0:
        # write down a list of rendered outputs in eval_dir
        for img_name in files[0]['names']:
            os.makedirs(osp.join(eval_dir, img_name), exist_ok=True)
        for idx, file in enumerate(files):
            for name, img in zip(file['names'], file['imgs']):
                img_path = osp.join(eval_dir, name, 'eval_{:04d}.png'.format(idx))
                cv2.imwrite(img_path, img)
        logger.add_log('Visual results add to {}'.format(eval_dir))

    logger.add_log('Evaluation Benchmark result. \n {}'.format(metric_info))

    # write to log file as well
    eval_log_file = os.path.join(cfgs.dir.eval_dir, 'eval_log.txt')
    with open(eval_log_file, 'w') as f:
        f.writelines('Model path {}\n'.format(cfgs.model_pt))
        f.writelines(metric_info)
