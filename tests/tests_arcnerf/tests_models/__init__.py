# -*- coding: utf-8 -*-

from thop import profile

from common.utils.cfgs_utils import obj_to_dict


def log_model_info(logger, model, feed_in, cfgs, batch_size, n_rays):
    # log model information
    logger.add_log('Model Layers:')
    logger.add_log(model)
    logger.add_log('')
    logger.add_log('Model Parameters: ')
    for n, _ in model.named_parameters():
        logger.add_log('   ' + n)
    flops, params = profile(model, inputs=(feed_in, ), verbose=False)
    logger.add_log('')
    logger.add_log('Model cfgs: ')
    log_nested_dict(logger, obj_to_dict(cfgs.model), extra='    ')
    logger.add_log('')
    logger.add_log('Module Flops/Params: ')
    logger.add_log('   Batch size: {}'.format(batch_size))
    logger.add_log('   N_rays: {}'.format(n_rays))
    logger.add_log('')
    if flops > 1024**3:
        flops, unit = flops / (1024.0**3), 'G'
    else:
        flops, unit = flops / (1024.0**2), 'M'
    logger.add_log('   Flops: {:.2f}{}'.format(flops, unit))
    logger.add_log('   Params: {:.2f}M'.format(params / (1024.0**2)))


def log_nested_dict(logger, nested_dict, extra=''):
    for k in nested_dict.keys():
        if isinstance(nested_dict[k], dict):
            log_nested_dict(logger, nested_dict[k], extra=extra + '    ')
        else:
            logger.add_log(extra + '{}: {}'.format(k, nested_dict[k]))
