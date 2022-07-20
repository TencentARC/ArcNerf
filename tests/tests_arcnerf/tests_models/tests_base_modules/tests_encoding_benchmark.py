#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import torch

from arcnerf.geometry.transformation import normalize
from arcnerf.models.base_modules.encoding import (
    DenseGridEmbedder, FreqEmbedder, GaussianEmbedder, HashGridEmbedder, SHEmbedder
)
from common.utils.logger import Logger
from tests.tests_arcnerf.tests_ops import get_start_time, get_end_time

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_rays = 4096
        cls.n_pts_per_rays = 64
        cls.logger = Logger(path=osp.join(RESULT_DIR, './benchmark.txt'), keep_console=False)
        cls.n_run = 100  # repeat the run to get avg

        cls.pts = torch.rand(cls.n_rays, cls.n_pts_per_rays, 3).view(-1, 3)  # (BN, 3)
        cls.dirs = torch.rand(cls.n_rays, cls.n_pts_per_rays, 3).view(-1, 3)  # (BN, 3)
        cls.dirs = normalize(cls.dirs)

    @staticmethod
    def to_cuda(item):
        """Move model or tensor to cuda"""
        if torch.cuda.is_available():
            item = item.cuda()

        return item

    def get_avg_time(self, model, input):
        t_forward, t_backward, t_forward_only = 0.0, 0.0, 0.0
        for _ in range(self.n_run):
            # forward
            t0 = get_start_time()
            output = model(input)
            t_forward += get_end_time(t0)

            # get loss
            loss = ((1 - output)**2).sum()

            # backward
            t0 = get_start_time()
            loss.backward()
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

    def log_time(self, model_name, t_f, t_b, t_f_o, dim):
        self.logger.add_log('_' * 60)
        self.logger.add_log('Model: {}'.format(model_name))
        self.logger.add_log('Input dim: {}'.format(dim))
        self.logger.add_log('   Forward: {:.6f}s'.format(t_f))
        self.logger.add_log('   Backward: {:.6f}s'.format(t_b))
        self.logger.add_log('   Forward-Only: {:.6f}s'.format(t_f_o))
        self.logger.add_log('_' * 60)

    def tests_freq_embedder(self):
        model = FreqEmbedder(3, 10, include_input=True)
        model = self.to_cuda(model)
        pts = self.to_cuda(self.pts.clone().requires_grad_(True))
        # single pts per ray
        pts_single = pts[:self.n_rays]
        t_f, t_b, t_f_o = self.get_avg_time(model, pts_single)
        self.log_time('Freq Embedder(Freq=10): ', t_f, t_b, t_f_o, dim=pts_single.shape)
        # full ray process
        t_f, t_b, t_f_o = self.get_avg_time(model, pts)
        self.log_time('Freq Embedder(Freq=10): ', t_f, t_b, t_f_o, dim=pts.shape)

    def tests_gaussian_embedder(self):
        model = GaussianEmbedder(3, 16, include_input=True)
        model = self.to_cuda(model)
        mean_cov = torch.cat([self.pts.clone()] * 2, dim=-1)
        mean_cov = self.to_cuda(mean_cov.requires_grad_(True))
        # single pts per ray
        mean_cov_single = mean_cov[:self.n_rays]
        t_f, t_b, t_f_o = self.get_avg_time(model, mean_cov_single)
        self.log_time('Gaussian Embedder(Freq=16): ', t_f, t_b, t_f_o, dim=mean_cov_single.shape)
        # full ray process
        t_f, t_b, t_f_o = self.get_avg_time(model, mean_cov)
        self.log_time('Gaussian Embedder(Freq=16): ', t_f, t_b, t_f_o, dim=mean_cov.shape)

    def tests_densegrid_encoder(self):
        model = DenseGridEmbedder(3, grid=256, include_input=True, W_feat=0)
        model = self.to_cuda(model)
        pts = self.to_cuda(self.pts.clone().requires_grad_(True))
        # single pts per ray
        pts_single = pts[:self.n_rays]
        t_f, t_b, t_f_o = self.get_avg_time(model, pts_single)
        self.log_time('DenseGrid Embedder(Grid=256, W_feat=0): ', t_f, t_b, t_f_o, dim=pts_single.shape)
        # full ray process
        t_f, t_b, t_f_o = self.get_avg_time(model, pts)
        self.log_time('DenseGrid Embedder(Grid=256, W_feat=0): ', t_f, t_b, t_f_o, dim=pts.shape)

    def tests_densegrid_encoder_W256(self):
        model = DenseGridEmbedder(3, grid=256, include_input=True, W_feat=256)
        model = self.to_cuda(model)
        pts = self.to_cuda(self.pts.clone().requires_grad_(True))
        # single pts per ray
        pts_single = pts[:self.n_rays]
        t_f, t_b, t_f_o = self.get_avg_time(model, pts_single)
        self.log_time('DenseGrid Embedder(Grid=256, W_feat=256): ', t_f, t_b, t_f_o, dim=pts_single.shape)
        # full ray process
        t_f, t_b, t_f_o = self.get_avg_time(model, pts)
        self.log_time('DenseGrid Embedder(Grid=256, W_feat=256): ', t_f, t_b, t_f_o, dim=pts.shape)

    def tests_sh_embedder_torch(self):
        model = SHEmbedder(3, 4, include_input=True)
        model = self.to_cuda(model)
        dirs = self.to_cuda(self.dirs.clone().requires_grad_(True))
        # single pts per ray
        dirs_single = dirs[:self.n_rays]
        t_f, t_b, t_f_o = self.get_avg_time(model, dirs_single)
        self.log_time('SH Embedder(Freq=4), torch-based: ', t_f, t_b, t_f_o, dim=dirs_single.shape)
        # full ray process
        t_f, t_b, t_f_o = self.get_avg_time(model, dirs)
        self.log_time('SH Embedder(Freq=4), torch-based:', t_f, t_b, t_f_o, dim=dirs.shape)

    def tests_sh_embedder_cuda(self):
        model = SHEmbedder(3, 4, include_input=True, use_cuda_backend=True)
        model = self.to_cuda(model)
        dirs = self.to_cuda(self.dirs.clone().requires_grad_(True))
        # single pts per ray
        dirs_single = dirs[:self.n_rays]
        t_f, t_b, t_f_o = self.get_avg_time(model, dirs_single)
        self.log_time('SH Embedder(Freq=4), cuda-based: ', t_f, t_b, t_f_o, dim=dirs_single.shape)
        # full ray process
        t_f, t_b, t_f_o = self.get_avg_time(model, dirs)
        self.log_time('SH Embedder(Freq=4), cuda-based:', t_f, t_b, t_f_o, dim=dirs.shape)

    def tests_hashgrid_embedder_torch(self):
        model = HashGridEmbedder(3, include_input=True)
        model = self.to_cuda(model)
        pts = self.to_cuda(self.pts.clone().requires_grad_(True))
        # single pts per ray
        pts_single = pts[:self.n_rays]
        t_f, t_b, t_f_o = self.get_avg_time(model, pts_single)
        self.log_time('HashGrid Embedder, torch-based: ', t_f, t_b, t_f_o, dim=pts_single.shape)
        # full ray process
        t_f, t_b, t_f_o = self.get_avg_time(model, pts)
        self.log_time('HashGrid Embedder, torch-based:', t_f, t_b, t_f_o, dim=pts.shape)

    def tests_hashgrid_embedder_cuda(self):
        model = HashGridEmbedder(3, include_input=True, use_cuda_backend=True)
        model = self.to_cuda(model)
        pts = self.to_cuda(self.pts.clone().requires_grad_(True))
        # single pts per ray
        pts_single = pts[:self.n_rays]
        t_f, t_b, t_f_o = self.get_avg_time(model, pts_single)
        self.log_time('HashGrid Embedder, cuda-based: ', t_f, t_b, t_f_o, dim=pts_single.shape)
        # full ray process
        t_f, t_b, t_f_o = self.get_avg_time(model, pts)
        self.log_time('HashGrid Embedder, cuda-based:', t_f, t_b, t_f_o, dim=pts.shape)
