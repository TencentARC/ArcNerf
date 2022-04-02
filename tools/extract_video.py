#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp

from common.utils.cfgs_utils import parse_configs
from common.utils.img_utils import get_n_img_in_dir, get_image_metadata
from common.utils.logger import Logger
from common.utils.video_utils import get_video_metadata, extract_video

if __name__ == '__main__':
    # parse args, logger
    cfgs = parse_configs()
    logger = Logger()

    video_path = cfgs.data.video_path
    logger.add_log('Start to extract images from video. Video path: {}'.format(video_path))
    assert osp.exists(video_path), 'No video file for processing...'

    scene_name = cfgs.data.scene_name
    scene_dir = osp.join(cfgs.dir.data_dir, 'Capture', scene_name)
    if osp.isdir(scene_dir):
        logger.add_log('Already exist {}. Do not process...'.format(scene_dir))
        exit()

    logger.add_log('Write to directory {}'.format(scene_dir))
    scene_img_dir = osp.join(scene_dir, 'images')
    os.mkdir(scene_dir)
    os.mkdir(scene_img_dir)

    num_frame, width, height, _ = get_video_metadata(video_path)
    logger.add_log('    Original video information: num_frame-{}, shape-{}/{}(w/h)'.format(num_frame, width, height))
    logger.add_log(
        '    Video Downsample: {}, Image Downsample {}'.format(cfgs.data.video_downsample, cfgs.data.image_downsample)
    )

    extract_video(
        video_path,
        scene_img_dir,
        video_downsample=cfgs.data.video_downsample,
        image_downsample=cfgs.data.image_downsample
    )
    logger.add_log('    Total image number extract: {}'.format(get_n_img_in_dir(scene_img_dir)))
    img_w, img_h, img_c = get_image_metadata(osp.join(scene_img_dir, '{:06d}.png'.format(0)))
    logger.add_log('    Extract image shape: {}/{}(w/h)'.format(img_w, img_h))
