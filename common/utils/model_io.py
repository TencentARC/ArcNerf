# -*- coding: utf-8 -*-

import os.path as osp

import torch


def load_model(logger, model, optimizer, path, cfgs, strict=False):
    """Load model from resume path. For distributed-gpu, it will have `module.` at the start of name"""
    # trained weights
    checkpoint = torch.load(path, map_location='cpu')

    # used for partial initialization
    input_dict = checkpoint['state_dict']
    state_dict = input_dict.copy()  # For param here, will not have module

    curr_dict = model.state_dict()
    # distributed model have params with name `module.xxx`
    if cfgs.dist.world_size > 1:
        state_dict_ = {}
        for k in state_dict.keys():
            new_key = 'module.' + k
            state_dict_[new_key] = state_dict[k]
    else:
        state_dict_ = state_dict

    for key in input_dict:
        if key not in curr_dict:
            continue

        if curr_dict[key].shape != input_dict[key].shape:
            state_dict_.pop(key)
            logger.add_log('key {} skipped because of size mismatch.'.format(key), level='warning')
    model.load_state_dict(state_dict_, strict=strict)

    # only load optimize in resume mode
    if optimizer is not None and 'optimizer' in checkpoint:
        if hasattr(cfgs.progress, 'start_epoch') and cfgs.progress.start_epoch < 0:
            optimizer.load_state_dict(checkpoint['optimizer'])

    # resume mode will start from current epoch. fine-tune will start from 0 or specify mode.
    keep_train = 'Load model only...'
    if hasattr(cfgs.progress, 'start_epoch'):
        keep_train = 'False(Start from {})'.format(cfgs.progress.start_epoch)
        if cfgs.progress.start_epoch < 0:
            cfgs.progress.start_epoch = max(0, checkpoint['epoch'])
            keep_train = 'True(Start from {})'.format(cfgs.progress.start_epoch)

    msg_load = 'Successfully loaded checkpoint from {} (at epoch {})... Keep Train: {}'.format(
        path, checkpoint['epoch'], keep_train
    )
    logger.add_log(msg_load)

    return model


def save_model(logger, model, optimizer, epoch, loss, model_dir, cfgs, spec_name=None):
    """Save model to model directory"""
    if cfgs.dist.world_size > 1:
        model_ = model.module
    else:
        model_ = model
    state_dict = model_.state_dict()
    # save checkpoint
    model_optim_state = {
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }

    if spec_name is not None:
        model_name = osp.join(model_dir, '{}.pt.tar'.format(spec_name))
    else:
        model_name = osp.join(model_dir, 'model_epoch{:03d}.pt.tar'.format(epoch))

    torch.save(model_optim_state, model_name)
    logger.add_log('Saved model at {} ...'.format(model_name))

    return model_name
