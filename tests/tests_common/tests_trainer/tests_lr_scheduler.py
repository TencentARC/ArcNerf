#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import torch
from torchvision.models import AlexNet

from common.trainer.lr_scheduler import get_learning_rate_scheduler
from common.visual.plot_2d import draw_2d_components
from tests import setup_test_config

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a dummy config"""
        cls.cfgs = setup_test_config()

    def tests_lr_scheduler(self):
        """Test lr scheduler and get visual results"""
        lr_scheduler_type = ['MultiStepLR', 'ExponentialLR', 'PolyLR', 'CosineAnnealingLR', 'WarmUpCosineLR']
        model = AlexNet(num_classes=1)
        lr = self.cfgs.optim.lr
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': lr}], lr=lr)

        lines = []
        legends = []
        for lr_type in lr_scheduler_type:
            self.cfgs.optim.lr_scheduler.type = lr_type

            lr_scheduler = get_learning_rate_scheduler(
                optimizer, last_epoch=-1, total_epoch=self.cfgs.progress.epoch, **self.cfgs.optim.lr_scheduler.__dict__
            )

            x = list(range(self.cfgs.progress.epoch))
            y = [self.cfgs.optim.lr]
            for _ in range(self.cfgs.progress.epoch - 1):
                optimizer.zero_grad()
                optimizer.step()
                lr = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()
                y.append(lr)
            lines.append([x, y])
            legends.append(lr_type)

        # write down visual 2d plot into result folder
        file_name = osp.join(RESULT_DIR, 'lr_scheduler.png')
        draw_2d_components(
            lines=lines, legends=legends, xlabel='step', ylabel='lr', title='lr scheduler', save_path=file_name
        )


if __name__ == '__main__':
    unittest.main()
