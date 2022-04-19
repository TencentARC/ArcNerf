# -*- coding: utf-8 -*-

import math
import random
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

        # cfgs for n_rays in training
        self.train_count = 0
        self.total_samples = None
        self.n_rays = get_value_from_cfgs_field(self.cfgs, 'n_rays', 1024)
        self.logger.add_log('Num of rays for each training batch: {}'.format(self.n_rays))

        data = {}
        # train
        data['train'] = self.set_train_dataset()

        # val. Only eval one sample per epoch.
        tkwargs_val = {'batch_size': 1, 'num_workers': self.cfgs.worker, 'pin_memory': True, 'drop_last': True}
        if not valid_key_in_cfgs(self.cfgs.dataset, 'val'):
            data['val'] = None
        else:
            data['val'], data['val_sampler'] = self.set_dataset('val', tkwargs_val)

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

    def set_train_dataset(self, epoch=0):
        """We will call this every time all rays have been trained.
        Every time we reset the dataset, make train_count as 0. And reset sampler by epoch
        """
        assert self.cfgs.dataset.train is not None, 'Please input train dataset...'
        tkwargs = {
            'batch_size': 1,  # does not matter
            'num_workers': self.cfgs.worker,
            'pin_memory': True,
            'drop_last': True
        }
        loader, sampler = self.set_dataset('train', tkwargs)
        if sampler is not None:  # in case sample can not be distributed equally among machine, do not want miss
            self.reset_sampler(sampler, epoch)

        # concat all batch
        self.train_count = 0
        data = self.concat_train_batch(loader)

        return data

    def concat_train_batch(self, loader):
        """Concat all elements from different samples together. The first dim is n_img * n_rays_per_img"""
        potential_keys = ['img', 'mask', 'rays_o', 'rays_d', 'bounds']

        concat_data = {'H': [0], 'W': [0]}  # will not write progress image during training
        total_item = len(loader)
        all_item = []
        for data in loader:
            all_item.append(data)

        rand_idx = list(range(total_item))
        random.shuffle(rand_idx)
        init_idx = rand_idx[0]
        for key in potential_keys:
            if key in all_item[init_idx]:
                concat_data[key] = all_item[init_idx][key]

        for idx in rand_idx[1:]:
            for key in potential_keys:
                if key in all_item[idx]:
                    concat_data[key] = torch.cat([concat_data[key], all_item[idx][key]], dim=0)

        for k, v in concat_data.items():
            if isinstance(v, torch.FloatTensor):
                v_shape = v.shape
                if self.total_samples is None:
                    self.total_samples = v_shape[0] * v_shape[1]
                else:  # TODO: do not support point cloud or other tensor not in (nhw, ...) now
                    assert self.total_samples == v_shape[0] * v_shape[1], 'Invalid input dim...'
                concat_data[k] = v.view(v_shape[0] * v_shape[1], *v_shape[2:])[None, :]  # (1, n_img * n_rays, ...)

        self.logger.add_log(
            'Need {} epoch to run all the rays...'.format(math.ceil(float(self.total_samples) / float(self.n_rays)))
        )

        return concat_data

    def get_train_batch(self):
        """Get the train batch base on self.train_count"""
        assert self.train_count < self.total_samples, 'All rays have been sampled, please reset train dataset...'

        data_batch = {}
        for k, v in self.data['train'].items():
            if isinstance(v, torch.FloatTensor):
                data_batch[k] = v[:, self.train_count:self.train_count + self.n_rays, ...]
            else:
                data_batch[k] = v

        self.train_count += self.n_rays

        return data_batch

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
        potential_keys = ['img', 'mask', 'rays_o', 'rays_d', 'bounds']
        feed_in = {}
        for key in potential_keys:
            if key in inputs:
                feed_in[key] = inputs[key]
                if device == 'gpu':
                    feed_in[key] = feed_in[key].cuda(non_blocking=True)

        # img must be there
        batch_size = inputs['img'].shape[0]

        return feed_in, batch_size

    @master_only
    def valid_epoch(self, epoch, step_in_epoch):
        """Validate the epoch.
           Remember to set eval mode at beginning and set train mode at the end.

           For object reconstruction, only one valid sample in each epoch. Shuffle sampler all the time.
        """
        self.logger.add_log('Valid on data...')

        # refresh valid sampler
        refresh = self.data['val_sampler'] is not None
        if refresh:
            self.reset_sampler(self.data['val_sampler'], epoch)

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

    def train_epoch(self, epoch):
        """Train for one epoch. Each epoch return the final sum of loss and total num of iter in epoch"""
        if epoch % self.cfgs.progress.epoch_loss == 0:
            self.logger.add_log('Epoch {:06d}:'.format(epoch))
        step_in_epoch = 1

        if self.train_count >= self.total_samples:
            self.logger.add_log('Reset training sample togethers... ')
            self.set_train_dataset(epoch)

        loss_all = self.train_step(epoch, 0, step_in_epoch, self.get_train_batch())

        self.lr_scheduler.step()

        return loss_all, step_in_epoch

    def train(self):
        """Train for the whole progress.

        In nerf-kind algorithm, it groups all rays from different samples together,
        then sample n_rays as the batch to the model. reset_sampler until all rays have been selected.
        """
        self.logger.add_log('-' * 60)
        if self.cfgs.progress.init_eval and self.data['eval'] is not None:
            self.eval_epoch(self.cfgs.progress.start_epoch)

        loss_all = 0.0
        for epoch in range(self.cfgs.progress.start_epoch, self.cfgs.progress.epoch):
            # train and record speed
            self.model.train()
            t_start = time.time()
            loss_all, step_in_epoch = self.train_epoch(epoch)
            epoch_time = time.time() - t_start
            if epoch % self.cfgs.progress.epoch_loss == 0:
                self.logger.add_log('Epoch time {:.3f} s/iter'.format(epoch_time))

            if (epoch + 1) % self.cfgs.progress.epoch_save_checkpoint == 0:
                self.save_model(epoch + 1, loss_all)

            if self.data['val'] is not None and self.cfgs.progress.epoch_val > 0:
                if epoch > 0 and epoch % self.cfgs.progress.epoch_val == 0:
                    self.valid_epoch(epoch, step_in_epoch)

            if self.data['eval'] is not None and self.cfgs.progress.epoch_eval > 0:
                if (epoch + 1) % self.cfgs.progress.epoch_eval == 0:
                    self.eval_epoch(epoch + 1)

        self.save_model(self.cfgs.progress.epoch, loss_all, spec_name='final')
        self.logger.add_log('Training for expr {} done... Final model is saved...'.format(self.cfgs.name))
