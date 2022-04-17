# -*- coding: utf-8 -*-

import time

import torch

from arcnerf.datasets import get_dataset
from arcnerf.datasets.transform.augmentation import get_transforms
from arcnerf.eval.eval_func import run_eval
from arcnerf.loss import build_loss
from arcnerf.metric import build_metric
from arcnerf.models import build_model
from arcnerf.visual.render_img import render_progress_img
from common.loss.loss_dict import LossDictCounter
from common.trainer.basic_trainer import BasicTrainer
from common.utils.cfgs_utils import valid_key_in_cfgs, get_value_from_cfgs_field
from common.utils.train_utils import master_only


class ArcNerfTrainer(BasicTrainer):
    """Trainer for Customized case"""

    def __init__(self, cfgs):
        super(ArcNerfTrainer, self).__init__(cfgs)

    def get_model(self):
        """Get custom model"""
        self.logger.add_log('-' * 60)
        model = build_model(self.cfgs, self.logger)

        return model

    def prepare_data(self):
        """Prepare dataset for train, val, eval. Gets data loader and sampler"""
        self.logger.add_log('-' * 60)
        data = {}
        # train
        assert self.cfgs.dataset.train is not None, 'Please input train dataset...'
        tkwargs = {
            'batch_size': self.cfgs.batch_size,
            'num_workers': self.cfgs.worker,
            'pin_memory': True,
            'drop_last': True
        }
        data['train'], data['train_sampler'] = self.set_dataset('train', tkwargs)

        # val. Only eval one sample per epoch.
        tkwargs['batch_size'] = 1
        if not valid_key_in_cfgs(self.cfgs.dataset, 'val'):
            data['val'] = None
        else:
            data['val'], data['val_sampler'] = self.set_dataset('val', tkwargs)

        # eval
        if not valid_key_in_cfgs(self.cfgs.dataset, 'eval'):
            data['eval'] = None
        else:
            eval_bs = get_value_from_cfgs_field(self.cfgs.dataset.eval, 'eval_batch_size', 1)
            tkwargs_eval = {
                'batch_size': eval_bs,
                'num_workers': self.cfgs.worker,
                'pin_memory': True,
                'drop_last': False
            }
            data['eval'], _ = self.set_dataset('eval', tkwargs_eval)

        return data

    def set_dataset(self, mode, tkwargs):
        """Get loader, sampler and aug_info"""
        transforms, _ = get_transforms(getattr(self.cfgs.dataset, mode))
        dataset = get_dataset(
            self.cfgs.dataset, self.cfgs.dir.data_dir, logger=self.logger, mode=mode, transfroms=transforms
        )

        sampler = None
        if mode != 'eval' and self.cfgs.dist.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.cfgs.dist.world_size, rank=self.cfgs.dist.rank
            )
        loader = torch.utils.data.DataLoader(dataset, sampler=sampler, shuffle=(sampler is None), **tkwargs)

        return loader, sampler

    def set_loss_factory(self):
        """Set loss factory which will be use to calculate all the loss"""
        self.logger.add_log('-' * 60)
        loss_factory = build_loss(self.cfgs, self.logger)

        return loss_factory

    def set_eval_metric(self):
        """Set eval metric which will be used for evaluation"""
        self.logger.add_log('-' * 60)
        eval_metric = build_metric(self.cfgs, self.logger)

        return eval_metric

    def get_model_feed_in(self, inputs, device):
        """Get the core model feed in and put it to the model's device"""
        feed_in = {'img': inputs['img'], 'mask': inputs['mask'], 'rays_o': inputs['rays_o'], 'rays_d': inputs['rays_d']}

        if device == 'gpu':
            for k in feed_in.keys():
                feed_in[k] = feed_in[k].cuda(non_blocking=True)

        batch_size = inputs['img'].shape[0]

        return feed_in, batch_size

    @master_only
    def valid_epoch(self, epoch, step_in_epoch):
        """Validate the epoch.
           Remember to set eval mode at beginning and set train mode at the end.

           For object reconstruction, only one valid sample in each epoch. Shuffle sampler all the time.
        """
        self.logger.add_log('Valid on data...')
        self.model.eval()
        loss_summary = LossDictCounter()
        count = 0
        global_step = (epoch + 1) * step_in_epoch
        for step, inputs in enumerate(self.data['val']):
            with torch.no_grad():
                feed_in, batch_size = self.get_model_feed_in(inputs, self.device)
                time0 = time.time()
                output = self.model(feed_in, get_progress=True)
                self.logger.add_log(
                    '   Valid one sample (H,W)=({},{}) time {:.2f}s'.format(
                        int(inputs['H'][0]), int(inputs['W'][0]),
                        time.time() - time0
                    )
                )
            if self.cfgs.progress.save_progress_val and step < self.cfgs.progress.max_samples_val:  # Just some samples
                self.save_progress(epoch, 0, global_step, inputs, output, mode='val')

            count += batch_size
            loss = self.calculate_loss(inputs, output)
            loss_summary(loss, batch_size)
            break  # only one batch per val epoch

        if count == 0:
            self.logger.add_log('Not batch was sent to valid...')
            self.model.train()
            return

        # get epoch average
        loss_summary.cal_average()

        if loss_summary.get_avg_summary() is not None:
            self.monitor.add_loss(loss_summary.get_avg_summary(), global_step, mode='val')
            loss_msg = 'Validation Avg Loss --> Sum [{:.2f}]'.format(loss_summary.get_avg_sum())
            self.logger.add_log(loss_msg)

        self.model.train()

    def render_progress_img(self, inputs, output):
        """Actual render for progress image with label. It is perform in each step with a batch.
         Return a dict with list of image and filename. filenames should be irrelevant to progress
         Image should be in bgr with shape(h,w,3), which can be directly writen by cv.imwrite().
         Return None will not write anything.
        """
        return render_progress_img(inputs, output)

    def evaluate(self, data, model, metric_summary, device, max_samples_eval):
        """Actual eval function for the model. Use run_eval since we want to run it locally as well"""
        metric_info, files = run_eval(
            data,
            self.get_model_feed_in,
            model,
            self.logger,
            self.eval_metric,
            metric_summary,
            device,
            self.render_progress_img,
            max_samples_eval,
            show_progress=False
        )

        return metric_info, files
