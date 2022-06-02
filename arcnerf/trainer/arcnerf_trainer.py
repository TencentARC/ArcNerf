# -*- coding: utf-8 -*-

import math
import os
import os.path as osp
import random
import time

import torch

from arcnerf.datasets import get_dataset, get_model_feed_in
from arcnerf.datasets.transform.augmentation import get_transforms
from arcnerf.eval.eval_func import run_eval
from arcnerf.eval.infer_func import set_inference_data, run_infer, write_infer_files
from arcnerf.loss import build_loss
from arcnerf.metric import build_metric
from arcnerf.models import build_model
from arcnerf.visual.plot_3d import draw_3d_components
from arcnerf.visual.render_img import render_progress_imgs, write_progress_imgs
from common.loss.loss_dict import LossDictCounter
from common.trainer.basic_trainer import BasicTrainer
from common.utils.cfgs_utils import valid_key_in_cfgs, get_value_from_cfgs_field
from common.utils.registry import LOSS_REGISTRY, METRIC_REGISTRY
from common.utils.torch_utils import torch_to_np
from common.utils.train_utils import master_only
from common.visual.plot_2d import draw_2d_components


class ArcNerfTrainer(BasicTrainer):
    """Trainer for Customized case"""

    def __init__(self, cfgs):
        # cfgs for n_rays in training
        self.train_count = 0
        self.total_samples = None
        self.shuffle_count = -1
        self.n_rays = 0
        # for importance sampling
        self.sample_mask = None
        self.sample_loss = None
        self.importance_sample = None

        # for inference only
        self.intrinsic = None
        self.wh = None
        # for progress save
        self.radius = None
        self.volume_dict = None

        super(ArcNerfTrainer, self).__init__(cfgs)
        self.get_progress = get_value_from_cfgs_field(self.cfgs.debug, 'get_progress', False)
        self.total_epoch = self.cfgs.progress.epoch

        # set eval metric during training
        self.train_metric = None
        if valid_key_in_cfgs(self.cfgs, 'train_metric'):
            self.train_metric = self.set_train_metric()

        # pretrain siren layer in implicit model
        self.logger.add_log('-' * 60)
        self.logger.add_log('Pretrain siren layers')
        self.model.pretrain_siren()

    def get_model(self):
        """Get custom model"""
        self.logger.add_log('-' * 60)
        model = build_model(self.cfgs, self.logger)

        return model

    def prepare_data(self):
        """Prepare dataset for train, val, eval, inference. Gets data loader and sampler"""
        self.logger.add_log('-' * 60)
        self.n_rays = get_value_from_cfgs_field(self.cfgs, 'n_rays', 1024)
        self.logger.add_log('Num of rays for each training batch: {}'.format(self.n_rays))

        data = {}
        # train
        data['train'] = self.set_train_dataset()
        # train data process cfgs
        data['train_scheduler'] = get_value_from_cfgs_field(self.cfgs.dataset.train, 'scheduler', None)

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

        # inference
        if valid_key_in_cfgs(self.cfgs, 'inference') and self.intrinsic is not None and self.wh is not None:
            self.logger.add_log('-' * 60)
            self.logger.add_log('Setting Inference data...')
            data['inference'] = set_inference_data(self.cfgs.inference, self.intrinsic, self.wh)

            if data['inference']['render'] is not None:
                self.logger.add_log(
                    'Render novel view - type: {}, n_cam {}, resolution: wh({}/{})'.format(
                        data['inference']['render']['cfgs']['type'], data['inference']['render']['cfgs']['n_cam'],
                        data['inference']['render']['wh'][0], data['inference']['render']['wh'][1]
                    )
                )
                if 'surface_render' in data['inference'] and data['inference']['surface_render'] is not None:
                    self.logger.add_log('Do surface rendering.')

            if data['inference']['volume'] is not data['inference']:
                self.logger.add_log(
                    'Extracting geometry from volume - n_grid {}'.format(data['inference']['volume']['cfgs']['n_grid'])
                )
        else:
            data['inference'] = None

        # set the sphere and volume for rendering
        if data['inference'] is not None:
            if data['inference']['render'] is not None:
                self.radius = data['inference']['render']['cfgs']['radius']
            if data['inference']['volume'] is not None:
                vol = data['inference']['volume']['Vol']
                self.volume_dict = {'grid_pts': torch_to_np(vol.get_corner()), 'lines': vol.get_bound_lines()}

        return data

    def set_train_dataset(self, epoch=0):
        """Set train dataset by collecting all rays together"""
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

        # concat all batch. Just do it once in beginning.
        data = self.concat_train_batch(loader)

        return data

    def concat_train_batch(self, loader):
        """Concat all elements from different samples together. The first dim is n_img * n_rays_per_img"""
        self.logger.add_log('Concat all training rays...')
        potential_keys = ['img', 'mask', 'rays_o', 'rays_d', 'bounds']

        concat_data = {}
        total_item = len(loader)
        # append all dict from different image
        all_item = []
        for data in loader:
            all_item.append(data)
            concat_data['H'] = int(data['H'][0])
            concat_data['W'] = int(data['W'][0])

        # randomly get the image sample idx
        rand_idx = list(range(total_item))
        random.shuffle(rand_idx)
        # init sample, just keep potential_keys
        init_idx = rand_idx[0]
        for key in potential_keys:
            if key in all_item[init_idx]:
                concat_data[key] = all_item[init_idx][key]
        # cat other sample tensors
        for idx in rand_idx[1:]:
            for key in potential_keys:
                if key in all_item[idx]:
                    concat_data[key] = torch.cat([concat_data[key], all_item[idx][key]], dim=0)

        # resize samples from (n_img, n_rays, ...) into (1, n_img * n_rays, ...)
        for k, v in concat_data.items():
            if isinstance(v, torch.Tensor):
                v_shape = v.shape
                # Any tensor will be in (n, hw, ...) shape
                # TODO: do not support point cloud or other tensor not in (nhw, ...) now
                if self.total_samples is None:
                    self.total_samples = v_shape[0] * v_shape[1]
                else:
                    assert self.total_samples == v_shape[0] * v_shape[1], 'Invalid input dim...'
                concat_data[k] = v.view(v_shape[0] * v_shape[1], *v_shape[2:])[None, :]  # (1, n_img * n_rays, ...)

        return concat_data

    def shuffle_train_data(self):
        """Shuffle all training data when total_count >= total_samples.
           Scheduler perform in each new shuffle.
        """
        self.train_count = 0
        self.shuffle_count += 1

        # this is the real train data to collect in each shuffle
        self.data['_train'] = {}

        # scheduler cfgs
        scheduler_cfg = self.data['train_scheduler']
        self.logger.add_log('-' * 60)
        self.logger.add_log('Shuffling training samples... Shuffle count - {}'.format(self.shuffle_count))

        # crop center image
        if scheduler_cfg is not None and valid_key_in_cfgs(scheduler_cfg, 'precrop') and \
                get_value_from_cfgs_field(scheduler_cfg.precrop, 'ratio', 1.0) < 1.0 and \
                self.shuffle_count <= get_value_from_cfgs_field(scheduler_cfg.precrop, 'max_shuffle', -1):
            keep_ratio = get_value_from_cfgs_field(scheduler_cfg.precrop, 'ratio', 1.0)
            self.logger.add_log('Crop training samples...keep ratio - {}'.format(keep_ratio))
            h, w = self.data['train']['H'], self.data['train']['W']
            for k, v in self.data['train'].items():
                if isinstance(v, torch.Tensor):
                    full_tensor = v.view(-1, h, w, *v.shape[2:])  # (N, H, W, ...)
                    dh, dw = int((1 - keep_ratio) * h / 2.0), int((1 - keep_ratio) * w / 2.0)
                    crop_tensor = full_tensor[:, dh:-dh, dw:-dw, ...]  # (N, H_c, W_c, ...)
                    self.total_samples = crop_tensor.shape[0] * crop_tensor.shape[1] * crop_tensor.shape[2]
                    self.data['_train'][k] = crop_tensor.reshape(1, self.total_samples, *crop_tensor.shape[3:])
        else:
            for k, v in self.data['train'].items():
                if isinstance(v, torch.Tensor):
                    self.data['_train'][k] = self.data['train'][k]  # get all (1, n_img*n_rays, ...)
                    self.total_samples = self.data['_train'][k].shape[1]

        # random shuffle of all remaining rays.
        if scheduler_cfg is None or get_value_from_cfgs_field(scheduler_cfg, 'random_shuffle', True):
            self.logger.add_log('Random shuffle all training samples...')
            random_idx = torch.randint(0, self.total_samples, size=[self.total_samples])
            for k, v in self.data['_train'].items():
                if isinstance(v, torch.Tensor):
                    self.data['_train'][k] = self.data['_train'][k][:, random_idx, ...]

        # importance sampling based on loss
        if scheduler_cfg is not None and valid_key_in_cfgs(scheduler_cfg, 'sample_loss') and \
                self.shuffle_count >= get_value_from_cfgs_field(scheduler_cfg.sample_loss, 'min_sample', -1):
            loss_keys = [k for k in list(scheduler_cfg.sample_loss.__dict__.keys()) if 'Loss' in k]
            sample_loss_key = loss_keys[0]  # you should have only one loss
            self.logger.add_log('Importance sampling on Loss {}'.format(sample_loss_key))
            self.sample_loss = LOSS_REGISTRY.get(sample_loss_key)(getattr(scheduler_cfg.sample_loss, sample_loss_key))

            if self.sample_mask is None or self.sample_mask.shape != (self.total_samples,) \
                    or not valid_key_in_cfgs(scheduler_cfg.sample_loss, 'sampling'):
                self.sample_mask = torch.zeros(self.total_samples)  # init a new loss mask
                self.importance_sample = torch.ones(self.total_samples).type(torch.BoolTensor)  # (n_img * n_rays)
            else:
                self.logger.add_log(
                    'Importance Sampling loss stat (min-{:.3f}/mean-{:.3f}/max-{:.3f})'.format(
                        self.sample_mask.min(), self.sample_mask.mean(), self.sample_mask.max()
                    )
                )
                # sample on all rays with large threshold, small error with random probability
                error_threshold = get_value_from_cfgs_field(scheduler_cfg.sample_loss.sampling, 'threshold', 0.0)
                random_ratio = get_value_from_cfgs_field(scheduler_cfg.sample_loss.sampling, 'random_ratio', 1.0)
                error_sample = torch.BoolTensor(self.sample_mask > error_threshold)  # (n_img * n_rays)
                random_sample = torch.logical_and(~error_sample, torch.rand(error_sample.shape) < random_ratio)
                self.importance_sample = torch.logical_or(error_sample, random_sample)  # (n_img * n_rays)
                # update the new selected batch
                for k, v in self.data['_train'].items():
                    if isinstance(v, torch.Tensor):
                        self.data['_train'][k] = self.data['_train'][k][:, self.importance_sample]
                        self.total_samples = self.data['_train'][k].shape[1]
                self.logger.add_log(
                    'Importance Sampling reduced sample num from {} to {}'.format(
                        self.sample_mask.shape[0], self.total_samples
                    )
                )

        self.logger.add_log(
            'Need {} epoch to run all the {} rays...'.format(
                math.ceil(float(self.total_samples) / float(self.n_rays)), self.total_samples
            )
        )
        self.logger.add_log('-' * 60)

        # set h/w as 0 to avoid writing of process image
        self.data['_train']['H'] = [0]
        self.data['_train']['W'] = [0]

    def get_train_batch(self):
        """Get the train batch base on self.train_count"""
        assert self.train_count < self.total_samples, 'All rays have been sampled, please reset train dataset...'

        data_batch = {}
        for k, v in self.data['_train'].items():
            if isinstance(v, torch.Tensor):  # tensor in (1, n_images * n_rays_per_image, ...)
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
        # must have eval data to set the intrinsic and wh
        if mode == 'eval':
            self.intrinsic = dataset.get_intrinsic(torch_tensor=False)
            self.wh = dataset.get_wh()

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

    def set_train_metric(self):
        """Set eval metric used in training"""
        self.logger.add_log('-' * 60)
        metric = [k for k in list(self.cfgs.train_metric.__dict__.keys())]
        train_metric_key = metric[0]  # you should have only one metric
        self.logger.add_log('Train Metric {}'.format(train_metric_key))
        train_metric = {
            'name': train_metric_key,
            'metric': METRIC_REGISTRY.get(train_metric_key)(getattr(self.cfgs.train_metric, train_metric_key))
        }

        return train_metric

    def set_eval_metric(self):
        """Set eval metric which will be used for evaluation"""
        self.logger.add_log('-' * 60)
        eval_metric = build_metric(self.cfgs, self.logger)

        return eval_metric

    def get_model_feed_in(self, inputs, device):
        """Get the core model feed in and put it to the model's device"""
        return get_model_feed_in(inputs, device)

    @master_only
    def train_step_writer(self, epoch, step, step_in_epoch, loss, learning_rate, global_step, inputs, output, **kwargs):
        """Write to monitor for saving, add params"""
        super().train_step_writer(
            epoch, step, step_in_epoch, loss, learning_rate, global_step, inputs, output, **kwargs
        )

        # write psnr
        if self.train_metric is not None:
            if epoch % self.cfgs.progress.epoch_loss == 0 and step % self.cfgs.progress.iter_loss == 0:
                metric_msg = 'Epoch {:06d} - Iter {}/{} - lr {:.8f}: '.format(
                    epoch, step, step_in_epoch - 1, learning_rate
                )
                metric = float(self.train_metric['metric'](inputs, output))
                metric_msg += '{} [{:.3f}] '.format(self.train_metric['name'], metric)
                self.logger.add_log(metric_msg)

                self.monitor.add_scalar(self.train_metric['name'], metric, global_step)

        # write params
        if 'params' in output:
            for k, v in output['params'][0].items():
                self.monitor.add_scalar(k, v, global_step, mode='train')

    def step_optimize(self, epoch, step, feed_in, inputs):
        """Set get progress for training"""
        output = self.model(feed_in, get_progress=self.get_progress, cur_epoch=epoch, total_epoch=self.total_epoch)
        loss = self.calculate_loss(inputs, output)

        # update loss by mask
        if self.sample_mask is not None:
            with torch.no_grad():
                loss_sample = self.sample_loss(inputs, output)
                reduce_dim = tuple(range(2, len(loss_sample.shape)))
                loss_sample = torch.mean(loss_sample, dim=reduce_dim)[0]  # (N_rays)
                update_idx = self.importance_sample.nonzero()[self.train_count - self.n_rays:self.train_count, 0]
                self.sample_mask[update_idx] = loss_sample.detach().cpu()

        self.optimizer.zero_grad()
        loss['sum'].backward()

        # grad clipping and step
        self.clip_gradients(epoch)
        self.optimizer.step()

        # print grad for debug
        self.debug_print_grad(epoch, step)

        return output, loss

    @master_only
    def valid_epoch(self, epoch, step_in_epoch):
        """Validate the epoch.
           Remember to set eval mode at beginning and set train mode at the end.

           For object reconstruction, only one valid sample in each epoch. Shuffle sampler all the time.
           get_progress for writing if debug.get_progress is True
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
        with torch.no_grad():
            for step, inputs in enumerate(self.data['val']):
                feed_in, batch_size = self.get_model_feed_in(inputs, self.device)
                time0 = time.time()
                output = self.model(feed_in, get_progress=self.get_progress)
                self.logger.add_log(
                    '   Valid one sample (H,W)=({},{}) time {:.2f}s'.format(
                        int(inputs['H'][0]), int(inputs['W'][0]),
                        time.time() - time0
                    )
                )
                if self.cfgs.progress.save_progress_val and step < self.cfgs.progress.max_samples_val:
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
            loss_msg = 'Validation Avg Loss --> Sum [{:.3f}]'.format(loss_summary.get_avg_sum())
            self.logger.add_log(loss_msg)

        # release gpu memory
        if self.device == 'gpu':
            torch.cuda.empty_cache()

        self.model.train()

    @master_only
    def infer_epoch(self, epoch):
        """Infer in this epoch, render path and extract mesh.
         Will be performed when dataset.eval is set and inference cfgs exists.
        """
        self.logger.add_log('Inference using model... Epoch {}'.format(epoch))
        eval_dir_epoch = osp.join(self.cfgs.dir.eval_dir, 'epoch_{:06d}'.format(epoch))
        os.makedirs(eval_dir_epoch, exist_ok=True)

        # inference and save result
        self.model.eval()
        files = self.inference(self.data['inference'], self.model, self.device)
        write_infer_files(files, eval_dir_epoch, self.data['inference'], self.logger)

        # release gpu memory
        if self.device == 'gpu':
            torch.cuda.empty_cache()

        self.model.train()

    def render_progress_imgs(self, inputs, output):
        """Actual render for progress image with label. It is perform in each step with a batch.
         Return a dict with list of image and filename. filenames should be irrelevant to progress
         Image should be in bgr with shape(h,w,3), which can be directly writen by cv.imwrite().
         Return None will not write anything.
        """
        return render_progress_imgs(inputs, output)

    @master_only
    def save_progress(self, epoch, step, global_step, inputs, output, mode='train'):
        """Save progress img for tracking. For both training and val. By default write to monitor.
            You are allowed to send a list of imgs to write for each iteration.
        """
        files = self.render_progress_imgs(inputs, output)
        if files is None:
            return

        # monitor write image
        if 'imgs' in files:
            for name, img in zip(files['imgs']['names'], files['imgs']['imgs']):
                self.monitor.add_img(name, img, global_step, mode=mode)
        # monitor write rays
        if 'rays' in files:
            if '2d' in files['rays']:
                for name, rays_2d in zip(files['rays']['2d']['names'], files['rays']['2d']['samples']):
                    fig = draw_2d_components(**rays_2d, title='2d rays, id: {}'.format(name), return_fig=True)
                    self.monitor.add_fig(name, fig, global_step, mode=mode)

            if '3d' in files['rays']:
                img = draw_3d_components(
                    **files['rays']['3d'],
                    sphere_radius=self.radius,
                    volume=self.volume_dict,
                    title='3d rays. pts size proportional to sigma',
                    plotly=True,
                    return_fig=True
                )
                self.monitor.add_img('rays_3d', img, global_step, mode=mode + '_rays3d')

        if self.cfgs.progress.local_progress:
            progress_dir = osp.join(self.cfgs.dir.expr_spec_dir, 'progress')
            os.makedirs(progress_dir, exist_ok=True)
            progress_mode_dir = osp.join(progress_dir, mode)
            os.makedirs(progress_mode_dir, exist_ok=True)
            self.write_progress_imgs([files], progress_mode_dir, epoch, step, global_step, False)

    def write_progress_imgs(self, files, folder, epoch=None, step=None, global_step=None, eval=False):
        """Actual function to write the progress images"""
        write_progress_imgs(files, folder, epoch, step, global_step, eval, self.radius, self.volume_dict)

    def inference(self, data, model, device):
        """Actual infer function for the model. Use run_infer since we want to run it local as well"""
        files = run_infer(data, self.get_model_feed_in, model, self.logger, device)

        return files

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
            self.render_progress_imgs,
            max_samples_eval,
            show_progress=False
        )

        return metric_info, files

    def train_epoch(self, epoch):
        """Train for one epoch. Each epoch return the final sum of loss and total num of iter in epoch"""
        step_in_epoch = 1  # in nerf, each epoch is sampling of all rays in all training samples

        if self.train_count >= self.total_samples:
            self.shuffle_train_data()

        loss_all = self.train_step(epoch, 0, step_in_epoch, self.get_train_batch())

        self.lr_scheduler.step()

        return loss_all, step_in_epoch

    def train(self):
        """Train for the whole progress.

        In nerf-kind algorithm, it groups all rays from different samples together,
        then sample n_rays as the batch to the model. reset_sampler until all rays have been selected.
        """
        self.logger.add_log('-' * 60)
        self.logger.add_log('Total num of epoch: {}'.format(self.cfgs.progress.epoch))

        # init eval/inference
        if self.cfgs.progress.init_eval and self.data['eval'] is not None:
            self.eval_epoch(self.cfgs.progress.start_epoch)
        if self.cfgs.progress.init_eval and self.data['inference'] is not None:
            self.infer_epoch(self.cfgs.progress.start_epoch)

        # init shuffle
        self.shuffle_train_data()

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

            if self.data['inference'] is not None and self.cfgs.progress.epoch_eval > 0:
                if (epoch + 1) % self.cfgs.progress.epoch_eval == 0:
                    self.infer_epoch(epoch + 1)

        self.save_model(self.cfgs.progress.epoch, loss_all, spec_name='final')
        self.logger.add_log('Training for expr {} done... Final model is saved...'.format(self.cfgs.name))
