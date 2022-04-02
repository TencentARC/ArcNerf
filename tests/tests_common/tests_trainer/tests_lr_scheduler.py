#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import torch
from torchvision.models import AlexNet

from common.trainer.lr_scheduler import get_learning_rate_scheduler
from common.visual.plot_2d import plot_curve_2d
from tests import setup_test_config

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfgs = setup_test_config()

    def tests_lr_scheduler(self):
        lr_scheduler_type = ['MultiStepLR', 'ExponentialLR', 'PolyLR', 'CosineAnnealingLR']
        model = AlexNet(num_classes=1)
        lr = self.cfgs.optim.lr
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': lr}], lr=lr)

        for lr_type in lr_scheduler_type:
            self.cfgs.optim.lr_scheduler.type = lr_type

            lr_scheduler = get_learning_rate_scheduler(
                optimizer, last_epoch=0, total_epoch=self.cfgs.progress.epoch, **self.cfgs.optim.lr_scheduler.__dict__
            )

            x = list(range(100))
            y = [self.cfgs.optim.lr]
            for epoch in range(99):
                optimizer.zero_grad()
                optimizer.step()
                lr = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()
                y.append(lr)

            file_name = osp.join(RESULT_DIR, 'lr_scheduler_{}.png'.format(lr_type))
            plot_curve_2d(x, y, 'epoch', 'lr', lr_type, save_path=file_name)
