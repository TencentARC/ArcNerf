#!/usr/bin/python
# -*- coding: utf-8 -*-

from . import TestModelDict


class TestDict(TestModelDict):

    def tests_volsdf_model(self):
        cfgs = self.load_model_configs('volsdf.yaml')
        logger = self.set_logger('volsdf.txt')
        model = self.build_model_to_cuda(cfgs, logger)

        feed_in = self.create_feed_in_to_cuda()
        self.log_model_info(logger, model, feed_in, cfgs)

        # test forward
        self._test_forward(model, feed_in, extra_keys=['normal'], extra_bn3=[True])

        # test params
        self._test_forward_params_in(model, feed_in, ['beta'])

        # inference only
        self._test_forward_inference_only(model, feed_in)

        # get progress
        n_sample = cfgs.model.rays.n_sample
        n_importance = (cfgs.model.rays.n_importance // cfgs.model.rays.n_iter) * cfgs.model.rays.n_iter
        n_total = n_sample + n_importance
        progress_shape = (self.batch_size, self.n_rays, n_total - 1)
        self._test_forward_progress(model, feed_in, progress_shape)

        # direct inference
        pts, view_dir = self.create_pts_dir_to_cuda()
        self._test_pts_dir_forward(model, pts, view_dir)
