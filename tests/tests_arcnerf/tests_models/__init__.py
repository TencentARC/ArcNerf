# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

from thop import profile
import torch

from arcnerf.models import build_model
from common.utils.cfgs_utils import obj_to_dict, load_configs, dict_to_obj
from common.utils.logger import log_nested_dict, Logger

CONFIG_DIR = osp.abspath(osp.join(__file__, '../../../..', 'configs', 'models'))
RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


def log_model_info(logger, model, feed_in, cfgs, batch_size, n_rays):
    """Forward inputs with (batch_size, n_rays, ...)"""
    # log model information
    logger.add_log('Model Layers:')
    logger.add_log(model)
    logger.add_log('')
    logger.add_log('Model Parameters: ')
    for n, _ in model.named_parameters():
        logger.add_log('   ' + n)
    flops, params = profile(model, inputs=(feed_in, ), verbose=False)
    logger.add_log('')
    logger.add_log('Model cfgs: ')
    log_nested_dict(logger, obj_to_dict(cfgs.model), extra='    ')
    logger.add_log('')
    logger.add_log('Module Flops/Params: ')
    logger.add_log('   Batch size: {}'.format(batch_size))
    logger.add_log('   N_rays: {}'.format(n_rays))
    logger.add_log('')
    if flops > 1024**3:
        flops, unit = flops / (1024.0**3), 'G'
    else:
        flops, unit = flops / (1024.0**2), 'M'
    logger.add_log('   Flops: {:.2f}{}'.format(flops, unit))
    logger.add_log('   Params: {:.2f}M'.format(params / (1024.0**2)))


def log_base_model_info(logger, model, feed_in, n_pts):
    """Forward inputs with (n_pts, ...)"""
    logger.add_log('Model Layers:')
    logger.add_log(model)
    logger.add_log('')
    logger.add_log('Model Parameters: ')
    for n, _ in model.named_parameters():
        logger.add_log('   ' + n)
    flops, params = profile(model, inputs=(feed_in, ), verbose=False)
    logger.add_log('Module Flops/Params: ')
    logger.add_log('   N_pts: {}'.format(n_pts))
    logger.add_log('')
    if flops > 1024**3:
        flops, unit = flops / (1024.0**3), 'G'
    else:
        flops, unit = flops / (1024.0**2), 'M'
    logger.add_log('   Flops: {:.2f}{}'.format(flops, unit))
    logger.add_log('   Params: {:.2f}M'.format(params / (1024.0**2)))


class TestModelDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 1
        cls.n_rays = 72 * 35

    def get_cfgs_logger(self, config_file, logger_file):
        cfgs = self.load_model_configs(config_file)
        logger = self.set_logger(logger_file)

        return cfgs, logger

    def load_model_configs(self, config_name):
        return load_configs(osp.join(osp.join(CONFIG_DIR, config_name)), None)

    def set_logger(self, logger_name):
        return Logger(path=osp.join(RESULT_DIR, logger_name), keep_console=False)

    @staticmethod
    def to_cuda(item):
        """Move model or tensor to cuda"""
        if torch.cuda.is_available():
            item = item.cuda()

        return item

    @staticmethod
    def add_model_field(cfgs):
        """Used for background models"""
        new_cfgs = {'model': obj_to_dict(cfgs)}

        return dict_to_obj(new_cfgs)

    def build_model_to_cuda(self, cfgs, logger):
        model = build_model(cfgs, logger)
        model = self.to_cuda(model)

        return model

    def create_feed_in_to_cuda(self):
        feed_in = {
            'img':
            torch.ones(self.batch_size, self.n_rays, 3),
            'mask':
            torch.ones(self.batch_size, self.n_rays),
            'rays_o':
            torch.rand(self.batch_size, self.n_rays, 3),
            'rays_d':
            torch.rand(self.batch_size, self.n_rays, 3),
            'near':
            torch.ones(self.batch_size * self.n_rays, 1) * 2.0,
            'far':
            torch.ones(self.batch_size * self.n_rays, 1) * 6.0,
            'bounds':
            torch.cat(
                [
                    torch.ones(self.batch_size, self.n_rays, 1) * 2.0,  # near
                    torch.ones(self.batch_size, self.n_rays, 1) * 6.0
                ],  # far
                dim=-1
            ),
        }
        for k, v in feed_in.items():
            feed_in[k] = self.to_cuda(v)

        return feed_in

    def create_pts_dir_to_cuda(self, num_pts=3):
        pts = torch.ones(self.n_rays, num_pts)
        view_dir = torch.ones(self.n_rays, 3)
        pts = self.to_cuda(pts)
        view_dir = self.to_cuda(view_dir)

        return pts, view_dir

    def log_model_info(self, logger, model, feed_in, cfgs):
        log_model_info(logger, model, feed_in, cfgs, self.batch_size, self.n_rays)

    def _test_pts_dir_forward(self, model, pts, view_dir):
        sigma = model.forward_pts(pts)
        self.assertEqual(sigma.shape, (self.n_rays, ))
        sigma, rgb = model.forward_pts_dir(pts, view_dir)
        self.assertEqual(sigma.shape, (self.n_rays, ))
        self.assertEqual(rgb.shape, (self.n_rays, 3))

    def _test_forward(self, model, feed_in, suffix='', extra_keys=None, extra_bn3=None):
        output = model(feed_in)
        self.assertEqual(output['rgb{}'.format(suffix)].shape, (self.batch_size, self.n_rays, 3))
        self.assertEqual(output['depth{}'.format(suffix)].shape, (
            self.batch_size,
            self.n_rays,
        ))
        self.assertEqual(output['mask{}'.format(suffix)].shape, (
            self.batch_size,
            self.n_rays,
        ))

        if extra_keys is not None:
            for idx, key in enumerate(extra_keys):
                if extra_bn3 is None:
                    gt_shape = (
                        self.batch_size,
                        self.n_rays,
                    )
                else:
                    gt_shape = (self.batch_size, self.n_rays, 3) if extra_bn3[idx] else (
                        self.batch_size,
                        self.n_rays,
                    )
                self.assertEqual(output['{}{}'.format(key, suffix)].shape, gt_shape)

    def _test_forward_inference_only(self, model, feed_in):
        """Test that all keys are not started with progress_"""
        output = model(feed_in, inference_only=True)
        self.assertEqual(output['rgb'].shape, (self.batch_size, self.n_rays, 3))
        self.assertTrue('rgb_coarse' not in output.keys())
        self.assertTrue('rgb_fine' not in output.keys())
        self.assertTrue(all([not k.startswith('progress_') for k in output.keys()]))

    def _test_forward_params_in(self, model, feed_in, params=None):
        """Test that param in forward output"""
        if params is not None:
            output = model(feed_in)
            for p in params:
                self.assertTrue(p in output['params'][0])

    def _test_forward_progress(self, model, feed_in, progress_shape):
        output = model(feed_in, get_progress=True)
        for key in ['sigma', 'zvals', 'alpha', 'trans_shift', 'weights']:
            self.assertEqual(output['progress_{}'.format(key)].shape, progress_shape)
        if model.sigma_reverse():
            self.assertTrue(output['progress_sigma_reverse'][0])

    def _test_surface_render(
        self, model, feed_in, method='sphere_tracing', grad_dir='ascent', extra_keys=None, extra_bn3=None
    ):
        output = model.surface_render(feed_in, method=method, grad_dir=grad_dir)

        if extra_keys is not None:
            for idx, key in enumerate(extra_keys):
                if extra_bn3 is None:
                    gt_shape = (
                        self.batch_size,
                        self.n_rays,
                    )
                else:
                    gt_shape = (self.batch_size, self.n_rays, 3) if extra_bn3[idx] else (
                        self.batch_size,
                        self.n_rays,
                    )
                self.assertEqual(output['{}'.format(key)].shape, gt_shape)
