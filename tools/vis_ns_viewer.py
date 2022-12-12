# -*- coding: utf-8 -*-

import time
from pathlib import Path
import os.path as osp

import torch

from arcnerf.datasets import get_dataset
from arcnerf.datasets.transform.augmentation import get_transforms
from arcnerf.models import build_model
from common.utils.logger import Logger
from common.utils.cfgs_utils import parse_configs, valid_key_in_cfgs, load_configs
from common.utils.model_io import load_model
from ns_viewer.server.viewer_utils import ViewerState
from ns_viewer.server.arcnerf_to_ns_viewer import arcnerf_dataset_to_ns_viewer


def main(dataset, logger, model, viewer_cfgs, log_base_dir: Path = Path('/tmp/nerfstudio_viewer_logs')):
    """Main function."""
    viewer_state = ViewerState(
        viewer_cfgs,
        logger,
        log_filename=log_base_dir / viewer_cfgs.relative_log_filename,
    )
    viewer_state.init_scene(dataset=dataset, start_train=False)
    logger.add_log('Please refresh and load page at: {}'.format(viewer_state.viewer_url))

    if model is None:
        time.sleep(30)  # allowing time to refresh page
    else:
        # keep rendering
        while True:
            viewer_state.vis['renderingState/isTraining'].write(False)
            update_viewer_state(viewer_state, model)


def update_viewer_state(viewer_state, model):
    """Update viewer by model"""
    try:
        viewer_state.update_scene(None, 1, model)
    except RuntimeError:
        time.sleep(0.03)  # sleep to allow buffer to reset


if __name__ == '__main__':
    # get config
    cfgs = parse_configs()
    if not valid_key_in_cfgs(cfgs, 'viewer'):
        proj_dir = osp.abspath(osp.join(__file__, '..', '..'))
        viewer_cfgs = load_configs(osp.join(proj_dir, 'configs', 'viewer.yaml')).viewer
    else:
        viewer_cfgs = cfgs.viewer

    # logger
    logger = Logger()

    # get dataset
    MODE = list(cfgs.dataset.__dict__.keys())[0]
    transforms, _ = get_transforms(getattr(cfgs.dataset, MODE))
    dataset = get_dataset(cfgs.dataset, cfgs.dir.data_dir, None, MODE, transforms)
    ns_dataset = arcnerf_dataset_to_ns_viewer(dataset)

    # model. Optional
    model = None
    if valid_key_in_cfgs(cfgs, 'model_pt') and osp.exists(cfgs.model_pt):
        assert torch.cuda.is_available(), 'Only render on GPU for online demo'
        model = build_model(cfgs, None)
        model = load_model(logger, model, None, cfgs.model_pt, cfgs)
        model.cuda()

    main(ns_dataset, logger, model, viewer_cfgs)
