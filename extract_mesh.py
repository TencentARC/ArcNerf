#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

import torch

from common.utils.cfgs_utils import parse_configs
from common.utils.logger import Logger
from common.utils.model_io import load_model

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
    assert cfgs.dir.eval_dir is not None, 'Please specify the eval_dir for saving inference results...'
    eval_dir = cfgs.dir.eval_dir
    os.makedirs(eval_dir, exist_ok=True)

    # set logger
    logger = Logger(rank=0)
    logger.add_log('Inference... Result write to {}...'.format(cfgs.dir.eval_dir))

    # set and load model
    assert cfgs.model_pt is not None, 'Please specify the model_pt for evaluation...'
    model = build_model(cfgs, logger)
    model = load_model(logger, model, None, cfgs.model_pt, cfgs)

    if device == 'gpu':
        model.cuda()
