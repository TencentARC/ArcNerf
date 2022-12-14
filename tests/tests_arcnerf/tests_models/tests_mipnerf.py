#!/usr/bin/python
# -*- coding: utf-8 -*-

from tests.tests_arcnerf.tests_models import TestModelDict


class TestNerfDict(TestModelDict):

    def tests_model(self):
        cfgs, logger = self.get_cfgs_logger('mipnerf.yaml', 'mipnerf.txt')
        model = self.build_model_to_cuda(cfgs, logger)

        feed_in = self.create_feed_in_to_cuda()
        self.log_model_info(logger, model, feed_in, cfgs)

        # test forward
        self._test_forward(model, feed_in, '_coarse')

        if cfgs.model.rays.n_importance > 0:
            self._test_forward(model, feed_in, '_fine')

        # inference only
        self._test_forward_inference_only(model, feed_in)

        # get progress
        n_sample = cfgs.model.rays.n_sample
        n_importance = cfgs.model.rays.n_importance
        add_inf_z = cfgs.model.rays.add_inf_z
        if n_importance > 0:
            progress_shape = (self.batch_size, self.n_rays, n_importance if add_inf_z else n_importance - 1)
        else:
            progress_shape = (self.batch_size, self.n_rays, n_sample if add_inf_z else n_sample - 1)
        self._test_forward_progress(model, feed_in, progress_shape)
