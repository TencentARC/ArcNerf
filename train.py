#!/usr/bin/python
# -*- coding: utf-8 -*-

from arcnerf.trainer.arcnerf_trainer import ArcNerfTrainer
from arcnerf.trainer.arcnerf_trainer_with_nsviewer import ArcNerfNSViewerTrainer
from common.utils.cfgs_utils import parse_configs, valid_key_in_cfgs

if __name__ == '__main__':
    # parse args
    cfgs = parse_configs()

    # trainer
    if valid_key_in_cfgs(cfgs, 'viewer'):
        trainer = ArcNerfNSViewerTrainer(cfgs)
    else:
        trainer = ArcNerfTrainer(cfgs)
    trainer.train()
