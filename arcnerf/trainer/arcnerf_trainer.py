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
from common.utils.torch_utils import torch_to_np
from common.utils.train_utils import master_only
from common.visual.plot_2d import draw_2d_components


class ArcNerfTrainer(BasicTrainer):
    """Trainer for Customized case"""

    def __init__(self, cfgs):
        super(ArcNerfTrainer, self).__init__(cfgs)
        self.get_progress = get_value_from_cfgs_field(self.cfgs.debug, 'get_progress', False)

    def get_model(self):
        """Get custom model"""
        self.logger.add_log('-' * 60)
        model = build_model(self.cfgs, self.logger)

        return model

    def prepare_data(self):
        """Prepare dataset for train, val, eval, inference. Gets data loader and sampler"""
        self.logger.add_log('-' * 60)

        # cfgs for n_rays in training
        self.train_count = 0
        self.total_samples = None
        self.n_rays = get_value_from_cfgs_field(self.cfgs, 'n_rays', 1024)
        self.logger.add_log('Num of rays for each training batch: {}'.format(self.n_rays))
        # for inference only
        self.intrinsic = None
        self.wh = None
        # for progress save
        self.radius = None
        self.volume_dict = None

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
            if data['inference']['volume'] is not data['inference']:
                self.logger.add_log(
                    'Extracting geometry from volume - n_grid {}'.format(data['inference']['volume']['cfgs']['n_grid'])
                )
        else:
            data['inference'] = None

        # set the sphere and volume for rendering
        if data['inference'] is not None and 'render' in data['inference']:
            self.radius = data['inference']['render']['cfgs']['radius']
        if data['inference'] is not None and 'volume' in data['inference']:
            vol = data['inference']['volume']['Vol']
            self.volume_dict = {'grid_pts': torch_to_np(vol.get_corner()), 'lines': vol.get_bound_lines()}

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
        data = self.concat_train_batch(loader)

        return data

    def concat_train_batch(self, loader):
        """Concat all elements from different samples together. The first dim is n_img * n_rays_per_img"""
        self.logger.add_log('Concat all training rays...')
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
            if isinstance(v, torch.Tensor):
                v_shape = v.shape
                if self.total_samples is None:
                    self.total_samples = v_shape[0] * v_shape[1]
                else:  # TODO: do not support point cloud or other tensor not in (nhw, ...) now
                    assert self.total_samples == v_shape[0] * v_shape[1], 'Invalid input dim...'
                concat_data[k] = v.view(v_shape[0] * v_shape[1], *v_shape[2:])[None, :]  # (1, n_img * n_rays, ...)

        return concat_data

    def shuffle_train_data(self):
        """Shuffle all training data"""
        self.train_count = 0

        self.logger.add_log('Shuffling training samples... ')
        random_idx = torch.randint(0, self.total_samples, size=[self.total_samples])
        for k, v in self.data['train'].items():
            if isinstance(v, torch.Tensor):
                self.data['train'][k] = self.data['train'][k][:, random_idx, ...]

        self.logger.add_log(
            'Need {} epoch to run all the rays...'.format(math.ceil(float(self.total_samples) / float(self.n_rays)))
        )

    def get_train_batch(self):
        """Get the train batch base on self.train_count"""
        assert self.train_count < self.total_samples, 'All rays have been sampled, please reset train dataset...'

        data_batch = {}
        for k, v in self.data['train'].items():
            if isinstance(v, torch.Tensor):
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

    def set_eval_metric(self):
        """Set eval metric which will be used for evaluation"""
        self.logger.add_log('-' * 60)
        eval_metric = build_metric(self.cfgs, self.logger)

        return eval_metric

    def get_model_feed_in(self, inputs, device):
        """Get the core model feed in and put it to the model's device"""
        return get_model_feed_in(inputs, device)

    def step_optimize(self, epoch, step, feed_in, inputs):
        """Set get progress for training"""
        output = self.model(feed_in, get_progress=self.get_progress)
        loss = self.calculate_loss(inputs, output)

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
           get_progress for writting if debug.get_progress is True
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
                output = self.model(feed_in, get_progress=self.get_progress)
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
                param = files['rays']['3d']
                if 'point_size' in param:
                    param['point_size'] = param['point_size'] / 5.0 if param['point_size'] is not None else None
                img = draw_3d_components(
                    **param,
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
        step_in_epoch = 1

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
