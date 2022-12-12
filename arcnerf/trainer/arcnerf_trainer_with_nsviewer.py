# -*- coding: utf-8 -*-

from pathlib import Path
import time
import sys

from arcnerf.datasets import get_dataset
from arcnerf.datasets.transform.augmentation import get_transforms
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.torch_utils import get_end_time, get_start_time
from ns_viewer.server.viewer_utils import ViewerState
from ns_viewer.server.arcnerf_to_ns_viewer import arcnerf_dataset_to_ns_viewer
from .arcnerf_trainer import ArcNerfTrainer


class ArcNerfNSViewerTrainer(ArcNerfTrainer):
    """Trainer for Customized case"""

    def __init__(self, cfgs):
        super(ArcNerfNSViewerTrainer, self).__init__(cfgs)

        # counter for time
        self.total_step_time = 0.0
        self.step_count = 0

        # setup viewer for train dataset. Only setup once
        transforms, _ = get_transforms(getattr(self.cfgs.dataset, 'train'))
        train_dataset = get_dataset(
            self.cfgs.dataset, self.cfgs.dir.data_dir, logger=self.logger, mode='train', transfroms=transforms
        )
        ns_dataset = arcnerf_dataset_to_ns_viewer(train_dataset)

        self.viewer_state = ViewerState(cfgs.viewer, self.logger, log_filename=Path('tmp/nerfstudio_viewer_logs'))
        self.viewer_state.init_scene(dataset=ns_dataset, start_train=False)
        self.viewer_state.vis['renderingState/isTraining'].write(True)

        self.logger.add_log('Please refresh and load page at: {}'.format(self.viewer_state.viewer_url))

    def train_step(self, epoch, step, step_in_epoch, inputs):
        """Train for one step. Rewrite for viewer setup"""
        loss_all = 0.0
        try:
            t_start = get_start_time()

            feed_in, _ = self.get_model_feed_in(inputs, self.device)
            output, loss = self.step_optimize(epoch, step, feed_in)
            loss_all = loss['sum']

            # add time
            self.total_step_time += get_end_time(t_start)
            self.step_count += 1
            avg_time = self.total_step_time / float(self.step_count)

            # for simplicity, we don't broadcast the loss from all device.
            learning_rate = self.lr_scheduler.get_last_lr()[0]
            global_step = step_in_epoch * epoch + step

            show_fg_only = get_value_from_cfgs_field(self.cfgs.viewer, 'show_fg_only', False)
            self.update_viewer_state(self.viewer_state, self.model, global_step, avg_time, show_fg_only)

            # write to monitor, include loss, output/gt visual
            self.train_step_writer(
                epoch, step, step_in_epoch, loss, learning_rate, global_step, feed_in, inputs, output
            )

            # save model after a period of time
            if self.cfgs.progress.save_time is not None and (self.cfgs.progress.save_time > 0) \
                    and (time.time() - self.time0 > self.cfgs.progress.save_time):
                self.logger.add_log('----Periodically save model now----')
                self.save_model(epoch + 1, loss_all, spec_name='latest')
                self.time0 = time.time()
                self.logger.add_log('----Periodically save model as latest.pt.tar----')

        except KeyboardInterrupt:  # not work for multiprocess launch
            self.save_model(epoch + 1, loss_all, spec_name='latest')
            self.logger.add_log('Keyboard Interrupt...', level='error')
            self.logger.add_log(
                'Training terminated at epoch {}... latest checkpoint saved...'.format(epoch), level='error'
            )
            sys.exit()

        return float(loss_all)

    def update_viewer_state(self, viewer_state, model, step, sec_per_batch, show_fg_only):
        """Update the viewer image"""
        # make the model render fg only
        fg_only = model.fg_only
        if show_fg_only:
            model.fg_only = True

        num_rays_per_batch = self.train_pipeline.get_info('n_rays')
        try:
            viewer_state.update_scene(None, step, model, num_rays_per_batch, sec_per_batch)
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset

        if show_fg_only:
            # Set back for training
            model.fg_only = fg_only
