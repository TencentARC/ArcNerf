#!/usr/bin/python
# -*- coding: utf-8 -*-

import os.path as osp
import subprocess

from arcnerf.colmap.colmap_func import estimate_poses, dense_reconstruct
from common.utils.cfgs_utils import parse_configs
from common.utils.file_utils import remove_dir_if_exists
from common.utils.logger import Logger

if __name__ == '__main__':
    # parse args, logger
    cfgs = parse_configs()
    logger = Logger()

    scene_name = cfgs.data.scene_name
    scene_dir = osp.join(cfgs.dir.data_dir, 'Capture', scene_name)
    if not osp.isdir(scene_dir):
        logger.add_log('{} does not exist. Extract video or put image first...'.format(scene_dir))
        exit()

    logger.add_log('Start to run COLMAP and estimate cam_poses... Scene dir: {}'.format(scene_dir))

    # run colmap by terminal
    try:
        estimate_poses(scene_dir, logger, cfgs.data.colmap.match_type)
    except subprocess.CalledProcessError:
        logger.add_log(
            'Can not run poses estimation. Check {} for detail...'.format(osp.join(scene_dir, 'colmap_output.txt')),
            level='Error'
        )
        remove_dir_if_exists(osp.join(scene_dir, 'sparse'))

    # run dense reconstruction by colmap. It takes time
    if cfgs.data.colmap.dense_reconstruct:
        try:
            dense_reconstruct(scene_dir, logger)
        except subprocess.CalledProcessError:
            logger.add_log(
                'Can not run dense reconstruction. Check {} for detail...'.format(
                    osp.join(scene_dir, 'colmap_dense_output.txt')
                ),
                level='Error'
            )
            remove_dir_if_exists(osp.join(scene_dir, 'dense'))
