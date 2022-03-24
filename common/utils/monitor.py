# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Monitor(object):
    """A monitor keeping track of all information during training"""

    def __init__(self, log_dir, rank=0):
        if rank != 0:
            raise RuntimeError('You should not set monitor on slaves...')

        self.log_dir = log_dir
        self.monitor = SummaryWriter(log_dir)

    def add_loss(self, loss, global_step, mode='train'):
        """Add a dict of loss"""
        for name in loss['names']:
            loss_name = '{}/{}'.format(str(mode), str(name))
            self.monitor.add_scalar(loss_name, float(loss[name]), global_step)
        self.monitor.add_scalar('{}/loss_sum'.format(str(mode)), float(loss['sum']), global_step)

    def add_scalar(self, key, value, global_step):
        """Add a scalar"""
        self.monitor.add_scalar(key, float(value), global_step)

    def add_img(self, filename, img, global_step, mode):
        """Add an image. Should be numpy array. """
        if len(img.shape) != 3:
            raise RuntimeError('Image shape error. ')

        if img.shape[-1] == 3 and img.shape[0] != 0:
            img = np.transpose(img, [2, 0, 1])

        self.monitor.add_image('{}/{}'.format(mode, filename), img, global_step)
