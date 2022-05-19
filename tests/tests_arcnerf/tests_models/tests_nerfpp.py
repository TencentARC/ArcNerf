#!/usr/bin/python
# -*- coding: utf-8 -*-

from tests.tests_arcnerf.tests_models import TestModelDict
from common.utils.cfgs_utils import get_value_from_cfgs_field


class TestNerfPPDict(TestModelDict):

    def tests_model(self):
        cfgs, logger = self.get_cfgs_logger('nerfpp.yaml', 'nerfpp.txt')
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
        n_total = n_sample + n_importance
        remove_last = 0
        if cfgs.model.background.bkg_blend == 'rgb':
            if get_value_from_cfgs_field(cfgs.model.rays, 'add_inf_z', False) is False:
                remove_last = 1

        if n_importance > 0:
            progress_shape = (self.batch_size, self.n_rays, n_total - remove_last)
        else:
            progress_shape = (self.batch_size, self.n_rays, n_sample - remove_last)
        self._test_forward_progress(model, feed_in, progress_shape)

        # direct pts/view
        pts, view_dir = self.create_pts_dir_to_cuda()
        self._test_pts_dir_forward(model, pts, view_dir)

        # surface render
        self._test_surface_render(model, feed_in, method='secant_root_finding', grad_dir='descent')
