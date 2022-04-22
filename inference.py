#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp

import torch

from common.utils.cfgs_utils import parse_configs
from common.utils.logger import Logger
from common.utils.model_io import load_model
from arcnerf.datasets import get_dataset, get_model_feed_in
from arcnerf.eval.infer_func import set_inference_data, run_infer, write_infer_files
from arcnerf.models import build_model

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
    logger = Logger(rank=0, path=osp.join(eval_dir, 'infer_log.txt'))
    logger.add_log('Inference on model... Result write to {}...'.format(cfgs.dir.eval_dir))

    # only for getting intrinsic, c2w. Use eval dataset for setting
    dataset = get_dataset(cfgs.dataset, cfgs.dir.data_dir, mode='eval', logger=logger)
    intrinsic = dataset.get_intrinsic(torch_tensor=False)
    wh = dataset.get_wh()

    # set and load model
    assert cfgs.model_pt is not None, 'Please specify the model_pt for evaluation...'
    model = build_model(cfgs, logger)
    model = load_model(logger, model, None, cfgs.model_pt, cfgs)

    if device == 'gpu':
        model.cuda()

    # prepare inference data
    logger.add_log('Setting Inference data...')
    data = set_inference_data(cfgs.inference, intrinsic, wh)
    if data is None:
        logger.add_log('You did not add any valid configs for inference, please check the configs...')
        exit()

    if data['render'] is not None:
        logger.add_log(
            'Render novel view - type: {}, n_cam {}, resolution: wh({}/{})'.format(
                data['render']['cfgs']['type'], data['render']['cfgs']['n_cam'], data['render']['wh'][0],
                data['render']['wh'][1]
            )
        )
    if data['volume'] is not None:
        logger.add_log('Extracting geometry from volume - n_grid {}'.format(data['volume']['cfgs']['n_grid']))

    # process data with model and get output
    files = run_infer(data, get_model_feed_in, model, logger, device)

    # write output
    write_infer_files(files, eval_dir, data, logger)
