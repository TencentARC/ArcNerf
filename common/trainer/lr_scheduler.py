# -*- coding: utf-8 -*-

import math

import numpy as np
import torch


def get_learning_rate_scheduler(
    optimizer,
    last_epoch=-1,
    total_epoch=100,
    type='MultiStepLR',
    lr_gamma=0.1,
    lr_steps=None,
    tmax=20,
    min_factor=0.1,
    **kwargs
):
    """Setup learning rate scheduler.
       Now support [MultiStepLR, ExponentialLR, PolyLR, CosineAnnealingLR, WarmUpCosineLR].
    """
    if lr_steps is None:
        lr_steps = []

    if type == 'ExponentialLR':
        lr_gamma_step = lr_gamma**(1.0 / lr_steps[0])
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma_step, last_epoch=last_epoch)
    elif type == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_steps, gamma=lr_gamma, last_epoch=last_epoch
        )
    elif type == 'PolyLR':
        lr_gamma = math.log(0.1) / math.log(1 - (lr_steps[0] - 1e-6) / total_epoch)

        # Poly with lr_gamma until args.lr_milestones[0], then stepLR with factor of 0.1
        def lambda_map(epoch_index):
            return math.pow(1 - epoch_index / total_epoch, lr_gamma) \
                if np.searchsorted(lr_steps, epoch_index + 1) == 0 \
                else math.pow(10, -1 * np.searchsorted(lr_steps, epoch_index + 1))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_map, last_epoch=last_epoch)
    elif type == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, last_epoch=last_epoch)
    elif type == 'WarmUpCosineLR':

        def lambda_map(epoch_index):
            if epoch_index < lr_steps[0]:
                learning_factor = epoch_index / lr_steps[0]
            else:
                progress = (epoch_index - lr_steps[0]) / (total_epoch - lr_steps[0])
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - min_factor) + min_factor
            return learning_factor

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_map, last_epoch=last_epoch)
    else:
        raise NameError('Unknown {} learning rate scheduler'.format(type))

    return lr_scheduler
