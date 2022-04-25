# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
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

    def add_img(self, filename, img, global_step, mode, bgr2rgb=True):
        """Add an image, color or grey scale are all allowed.
         Should be numpy array in (h, w, c). c is optional and equal to 1/3
         If bgr2rgb is True, will assume the c in bgr order and change it to rgb order

         Monitor takes (h, w) or (c, h, w) image for writing.
        """
        if len(img.shape) != 2 and len(img.shape) != 3:
            raise RuntimeError('Image shape error. ')

        if len(img.shape) == 3 and img.shape[-1] == 3:  # rgb scale
            if bgr2rgb:
                img = img[:, :, [2, 1, 0]]
            img = np.transpose(img, [2, 0, 1])

        if len(img.shape) == 3 and img.shape[-1] == 1:  # gray scale
            img = np.transpose(img, [2, 0, 1])

        self.monitor.add_image('{}/{}'.format(mode, filename), img, global_step)

    def add_fig(self, filename, fig, global_step, mode):
        """Add a fig from plt"""
        self.monitor.add_figure('{}_fig/{}'.format(mode, filename), fig, global_step)
        plt.close(fig)
