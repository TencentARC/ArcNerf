#!/usr/bin/python
# -*- coding: utf-8 -*-

from simplengp.trainer.simplengp_trainer import SimplengpTrainer
from common.utils.cfgs_utils import parse_configs

if __name__ == '__main__':
    # parse args
    cfgs = parse_configs()

    # trainer
    trainer = SimplengpTrainer(cfgs)
    trainer.train()
