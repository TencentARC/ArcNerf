#!/usr/bin/python
# -*- coding: utf-8 -*-

from common.utils.cfgs_utils import parse_configs
from simplenerf.trainer.simplenerf_trainer import SimplenerfTrainer

if __name__ == '__main__':
    # parse args
    cfgs = parse_configs()

    # trainer
    trainer = SimplenerfTrainer(cfgs)
    trainer.train()
