# -*- coding: utf-8 -*-

import os
import os.path as osp
import time

from thop import profile
import torch

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


def log_base_model_info(logger, model, feed_in, n_pts):
    """Forward inputs with (n_pts, ...)"""
    logger.add_log('Model Layers:')
    logger.add_log(model)
    logger.add_log('')
    logger.add_log('Model Parameters: ')
    for n, _ in model.named_parameters():
        logger.add_log('   ' + n)
    flops, params = profile(model, inputs=(feed_in, ), verbose=False)
    logger.add_log('Module Flops/Params: ')
    logger.add_log('   N_pts: {}'.format(n_pts))
    logger.add_log('')
    if flops > 1024**3:
        flops, unit = flops / (1024.0**3), 'G'
    else:
        flops, unit = flops / (1024.0**2), 'M'
    logger.add_log('   Flops: {:.2f}{}'.format(flops, unit))
    logger.add_log('   Params: {:.2f}M'.format(params / (1024.0**2)))

    if torch.cuda.is_available():
        model = model.cuda()
        feed_in = feed_in.cuda()
        torch.cuda.synchronize()
        time0 = time.time()
        _ = model(feed_in)
        torch.cuda.synchronize()
        logger.add_log('For {} pts time {:.5f}s'.format(n_pts, time.time() - time0))
