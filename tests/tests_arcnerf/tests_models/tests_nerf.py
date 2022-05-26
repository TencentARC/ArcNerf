#!/usr/bin/python
# -*- coding: utf-8 -*-

from tests.tests_arcnerf.tests_models import TestModelDict


class TestNerfDict(TestModelDict):

    def tests_model(self):
        # multiple choice
        model_cfgs = ['nerf.yaml', 'nerf_dv_withnn.yaml']
        model_logs = [name.split('.')[0] + '.txt' for name in model_cfgs]

        for model_cfg, model_log in zip(model_cfgs, model_logs):
            cfgs, logger = self.get_cfgs_logger(model_cfg, model_log)
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
            add_inf_z = cfgs.model.rays.add_inf_z
            if n_importance > 0:
                progress_shape = (self.batch_size, self.n_rays, n_total if add_inf_z else n_total - 1)
            else:
                progress_shape = (self.batch_size, self.n_rays, n_sample if add_inf_z else n_sample - 1)
            self._test_forward_progress(model, feed_in, progress_shape)

            # direct pts/view
            pts, view_dir = self.create_pts_dir_to_cuda()
            self._test_pts_dir_forward(model, pts, view_dir)

            # surface render
            self._test_surface_render(model, feed_in, method='secant_root_finding', grad_dir='descent')
