#!/usr/bin/python
# -*- coding: utf-8 -*-

from arcnerf.trainer.arcnerf_trainer import ArcNerfTrainer
from common.utils.cfgs_utils import parse_configs

if __name__ == '__main__':
    # parse args
    cfgs = parse_configs()

    # trainer
    trainer = ArcNerfTrainer(cfgs)
    trainer.train()
