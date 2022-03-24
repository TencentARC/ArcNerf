#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from torchvision.models import AlexNet

from common.trainer.lr_scheduler import get_learning_rate_scheduler
from common.utils.cfgs_utils import parse_configs
from common.visual.plot_2d import plot_curve_2d

if __name__ == '__main__':
    cfgs = parse_configs()

    model = AlexNet(num_classes=1)
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': cfgs.optim.lr}], lr=cfgs.optim.lr)

    print('Test lr_scheduler')
    print('Init lr is ', cfgs.optim.lr)
    print('lr_scheduler is ', cfgs.optim.lr_scheduler.type)
    print('Setting ', cfgs.optim.lr_scheduler.__dict__)

    lr_scheduler = get_learning_rate_scheduler(
        optimizer, last_epoch=0, total_epoch=cfgs.progress.epoch, **cfgs.optim.lr_scheduler.__dict__
    )

    x = list(range(100))
    y = [cfgs.optim.lr]
    for epoch in range(99):
        optimizer.zero_grad()
        optimizer.step()
        lr = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()
        y.append(lr)

    plot_curve_2d(x, y, 'epoch', 'lr', cfgs.optim.lr_scheduler.type)
