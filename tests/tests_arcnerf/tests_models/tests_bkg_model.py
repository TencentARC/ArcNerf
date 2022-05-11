#!/usr/bin/python
# -*- coding: utf-8 -*-

from . import TestModelDict


class TestDict(TestModelDict):

    def tests_nerfpp_model(self):
        cfgs = self.load_model_configs('nerfpp.yaml')
        cfgs = self.add_model_field(cfgs.model.background)
        logger = self.set_logger('bkg_nerfpp.txt')
        model = self.build_model_to_cuda(cfgs, logger)

        feed_in = self.create_feed_in_to_cuda()
        self.log_model_info(logger, model, feed_in, cfgs)

        # test forward
        self._test_forward(model, feed_in)

        # test inference only
        self._test_forward_inference_only(model, feed_in)

        # test_get_progress
        n_sample = cfgs.model.rays.n_sample
        progress_shape = (self.batch_size, self.n_rays, n_sample if cfgs.model.rays.add_inf_z else n_sample - 1)
        self._test_forward_progress(model, feed_in, progress_shape)

        # direct pts/view
        pts, view_dir = self.create_pts_dir_to_cuda(4)
        self._test_pts_dir_forward(model, pts, view_dir)
