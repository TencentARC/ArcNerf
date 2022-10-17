#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import torch

from arcnerf.geometry.transformation import normalize
from arcnerf.loss import build_loss
from arcnerf.models import build_model
from common.utils.cfgs_utils import load_configs, dict_to_obj
from common.utils.logger import Logger
from tests.tests_arcnerf.tests_ops import get_start_time, get_end_time

CONFIG_DIR = osp.abspath(osp.join(__file__, '../../../..', 'configs', 'models'))
RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 1
        cls.n_rays = 4096
        cls.logger = Logger(path=osp.join(RESULT_DIR, './benchmark.txt'), keep_console=False)
        cls.n_run = 100  # repeat the run to get avg
        cls.max_pos = 3.0

        # data
        rays_o = torch.rand(cls.batch_size, cls.n_rays, 3) * cls.max_pos
        rays_d = -normalize(rays_o + torch.rand_like(rays_o) * cls.max_pos)  # point to origin with noise
        rays_r = torch.rand(cls.batch_size, cls.n_rays, 1)
        bn3 = torch.ones(cls.batch_size, cls.n_rays, 3)
        bn1 = torch.ones(cls.batch_size, cls.n_rays, 1)
        bn = torch.ones(cls.batch_size, cls.n_rays)
        cls.data = {
            'img': bn3,
            'mask': bn,
            'rays_o': rays_o,
            'rays_d': rays_d,
            'rays_r': rays_r,
            'near': bn1 * 2.0,
            'far': bn1 * 6.0,
            'bounds': torch.cat([bn1 * 2.0, bn1 * 6.0], dim=-1)
        }

        for k, v in cls.data.items():
            cls.data[k] = cls.to_cuda(v)

        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        cls.logger.add_log('Device {}'.format(device))
        cls.logger.add_log('Num of run {}'.format(cls.n_run))
        cls.logger.add_log('Batch size {}'.format(cls.batch_size))
        cls.logger.add_log('Num of rays {}'.format(cls.n_rays))

    def load_cfgs(self, name):
        cfgs = load_configs(osp.join(CONFIG_DIR, name))

        return cfgs

    @staticmethod
    def to_cuda(item):
        """Move model or tensor to cuda"""
        if torch.cuda.is_available():
            item = item.cuda()

        return item

    def build_test_model(self, cfgs_name):
        return self.to_cuda(build_model(self.load_cfgs(cfgs_name), None))

    def build_loss_func(self, keys):
        """Build rgb loss with certain keys"""
        cfgs = {
            'loss': {
                'ImgLoss': {
                    'keys': keys,
                    'weight': 1.0
                },
            }
        }
        loss_func = build_loss(dict_to_obj(cfgs), None)

        return loss_func

    def get_avg_time(self, model, input, loss_func):
        t_forward, t_backward, t_forward_only = 0.0, 0.0, 0.0
        for _ in range(self.n_run):
            # clear the grad
            for _, v in input.items():
                if isinstance(v, torch.Tensor):
                    if v.grad is not None:
                        v.grad.zero_()

            # forward
            t0 = get_start_time()
            outputs = model(input)
            t_forward += get_end_time(t0)

            # get loss
            loss = loss_func(input, outputs)

            # backward
            t0 = get_start_time()
            loss['sum'].backward()
            t_backward += get_end_time(t0)

            # forward only
            with torch.no_grad():
                t0 = get_start_time()
                _ = model(input)
                t_forward_only += get_end_time(t0)

        t_forward = t_forward / float(self.n_run)
        t_backward = t_backward / float(self.n_run)
        t_forward_only = t_forward_only / float(self.n_run)

        return t_forward, t_backward, t_forward_only

    def log_time(self, model_name, t_f, t_b, t_f_o):
        self.logger.add_log('_' * 60)
        self.logger.add_log('Model: {}'.format(model_name))
        self.logger.add_log('   Forward: {:.6f}s'.format(t_f))
        self.logger.add_log('   Backward: {:.6f}s'.format(t_b))
        self.logger.add_log('   Forward-Only: {:.6f}s'.format(t_f_o))
        self.logger.add_log('_' * 60)

    def tests_nerf(self):
        cfgs_name = 'nerf.yaml'
        model = self.build_test_model(cfgs_name)
        loss_func = self.build_loss_func(['rgb_fine'])

        t_forward, t_backward, t_forward_only = self.get_avg_time(model, self.data, loss_func)
        self.log_time('NeRF ', t_forward, t_backward, t_forward_only)

    def tests_nerfpp(self):
        cfgs_name = 'nerfpp.yaml'
        model = self.build_test_model(cfgs_name)
        loss_func = self.build_loss_func(['rgb_fine'])

        t_forward, t_backward, t_forward_only = self.get_avg_time(model, self.data, loss_func)
        self.log_time('NeRF++ ', t_forward, t_backward, t_forward_only)

    def tests_neus(self):
        cfgs_name = 'neus.yaml'
        model = self.build_test_model(cfgs_name)
        loss_func = self.build_loss_func(['rgb'])

        t_forward, t_backward, t_forward_only = self.get_avg_time(model, self.data, loss_func)
        self.log_time('Neus ', t_forward, t_backward, t_forward_only)

    def tests_volsdf(self):
        cfgs_name = 'volsdf.yaml'
        model = self.build_test_model(cfgs_name)
        loss_func = self.build_loss_func(['rgb'])

        t_forward, t_backward, t_forward_only = self.get_avg_time(model, self.data, loss_func)
        self.log_time('Volsdf ', t_forward, t_backward, t_forward_only)

    def tests_mipnerf(self):
        cfgs_name = 'mipnerf.yaml'
        model = self.build_test_model(cfgs_name)
        loss_func = self.build_loss_func(['rgb_fine'])

        t_forward, t_backward, t_forward_only = self.get_avg_time(model, self.data, loss_func)
        self.log_time('mipnerf ', t_forward, t_backward, t_forward_only)

    def tests_nerf_ngp(self):
        cfgs_name = 'nerf_ngp.yaml'
        model = self.build_test_model(cfgs_name)
        loss_func = self.build_loss_func(['rgb_coarse'])

        t_forward, t_backward, t_forward_only = self.get_avg_time(model, self.data, loss_func)
        self.log_time('NeRF-ngp ', t_forward, t_backward, t_forward_only)
