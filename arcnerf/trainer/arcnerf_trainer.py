# -*- coding: utf-8 -*-

import os
import os.path as osp
import time

import torch

from .pipeline import Pipeline
from arcnerf.datasets import get_dataset, get_model_feed_in, POTENTIAL_KEYS
from arcnerf.datasets.transform.augmentation import get_transforms
from arcnerf.eval.eval_func import run_eval
from arcnerf.eval.infer_func import Inferencer
from arcnerf.loss import build_loss
from arcnerf.metric import build_metric
from arcnerf.models import build_model
from arcnerf.trainer.ema import EMA
from arcnerf.visual.plot_3d import draw_3d_components
from arcnerf.visual.render_img import render_progress_imgs, write_progress_imgs
from common.loss.loss_dict import LossDictCounter
from common.trainer.basic_trainer import BasicTrainer
from common.utils.cfgs_utils import valid_key_in_cfgs, get_value_from_cfgs_field
from common.utils.registry import METRIC_REGISTRY
from common.utils.train_utils import master_only
from common.visual.plot_2d import draw_2d_components


class ArcNerfTrainer(BasicTrainer):
    """Trainer for Customized case"""

    def __init__(self, cfgs):
        # train pipeline for handling the inputs
        self.train_pipeline = Pipeline()

        # for inference only
        self.intrinsic = None
        self.wh = None
        # for progress visual
        self.radius = None
        self.volume_dict = None

        # call parent init
        super(ArcNerfTrainer, self).__init__(cfgs)
        self.get_progress = get_value_from_cfgs_field(self.cfgs.debug, 'get_progress', False)
        self.total_epoch = self.cfgs.progress.epoch

        # set eval metric during training
        self.train_metric = None
        if valid_key_in_cfgs(self.cfgs, 'train_metric'):
            self.train_metric = self.set_train_metric()

        # pretrain siren layer in implicit model
        self.logger.add_log('-' * 60)
        self.logger.add_log('Init Setting of the model...')
        self.model.init_setting()

        # ema update
        self.ema = None
        if get_value_from_cfgs_field(self.cfgs.optim, 'ema') is not None:
            decay = get_value_from_cfgs_field(self.cfgs.optim.ema, 'decay', 0.95)
            self.ema = EMA(self.model, decay)
            self.ema.set_n_step(self.cfgs.progress.start_epoch)
            self.logger.add_log('-' * 60)
            self.logger.add_log('EMA Update with decay {}'.format(decay))

    def get_model(self):
        """Get custom model"""
        self.logger.add_log('-' * 60)
        model = build_model(self.cfgs, self.logger)

        return model

    def empty_gpu_cache(self):
        """Empty gpu cache is valid"""
        if self.device == 'gpu':
            torch.cuda.empty_cache()

    def prepare_data(self):
        """Prepare dataset for train, val, eval, inference. Gets data loader and sampler"""
        self.logger.add_log('-' * 60)
        self.train_pipeline.set_n_rays(self.logger, get_value_from_cfgs_field(self.cfgs, 'n_rays', 1024))

        data = {}
        # train
        data['train'] = self.set_train_dataset()
        # train data process cfgs
        self.train_pipeline.setup_cfgs(get_value_from_cfgs_field(self.cfgs.dataset.train, 'scheduler', None))

        # val. Only eval one sample per epoch.
        if not valid_key_in_cfgs(self.cfgs.dataset, 'val'):
            data['val'] = None
        else:
            data_on_gpu = (get_value_from_cfgs_field(self.cfgs.dataset.val, 'device', 'cpu') == 'gpu')
            tkwargs_val = {
                'batch_size': 1,
                'num_workers': self.cfgs.worker if not data_on_gpu else 0,
                'pin_memory': not data_on_gpu,
                'drop_last': True
            }
            data['val'], data['val_sampler'] = self.set_dataset('val', tkwargs_val)

        # eval
        if not valid_key_in_cfgs(self.cfgs.dataset, 'eval'):
            data['eval'] = None
        else:
            eval_bs = get_value_from_cfgs_field(self.cfgs.dataset.eval, 'eval_batch_size', 1)
            data_on_gpu = (get_value_from_cfgs_field(self.cfgs.dataset.eval, 'device', 'cpu') == 'gpu')
            tkwargs_eval = {
                'batch_size': eval_bs,
                'num_workers': self.cfgs.worker if not data_on_gpu else 0,
                'pin_memory': not data_on_gpu,
                'drop_last': False
            }
            data['eval'], _ = self.set_dataset('eval', tkwargs_eval)

        # inference
        if valid_key_in_cfgs(self.cfgs, 'inference') and self.intrinsic is not None and self.wh is not None:
            self.logger.add_log('-' * 60)
            self.logger.add_log('Setting Inference data...')
            to_gpu = get_value_from_cfgs_field(self.cfgs.inference, 'to_gpu', False)
            data['inference'] = Inferencer(
                self.cfgs.inference, self.intrinsic, self.wh, self.device, self.logger, to_gpu=to_gpu
            )

            if data['inference'].is_none():
                data['inference'] = None
            else:
                if data['inference'].get_render_data() is not None:
                    render_cfgs = data['inference'].get_render_cfgs()
                    wh = data['inference'].get_wh()
                    self.logger.add_log(
                        'Render novel view - type: {}, n_cam {}, resolution: wh({}/{})'.format(
                            render_cfgs['type'], render_cfgs['n_cam'], wh[0], wh[1]
                        )
                    )
                    if 'surface_render' in data['inference'].get_render_data() \
                            and data['inference'].get_render_data()['surface_render'] is not None:
                        self.logger.add_log('Do surface rendering.')

                if data['inference'].get_volume_data() is not None:
                    self.logger.add_log(
                        'Extracting geometry from volume - n_grid {}'.format(
                            data['inference'].get_volume_cfgs()['n_grid']
                        )
                    )

        else:
            data['inference'] = None

        # set the sphere and volume for rendering
        if data['inference'] is not None:
            self.radius = data['inference'].get_radius()
            self.volume_dict = data['inference'].get_volume_dict()

        self.empty_gpu_cache()

        return data

    def set_train_dataset(self, epoch=0):
        """Set train dataset by collecting all rays together"""
        self.logger.add_log('-' * 60)
        assert self.cfgs.dataset.train is not None, 'Please input train dataset...'

        data_on_gpu = (get_value_from_cfgs_field(self.cfgs.dataset.train, 'device', 'cpu') == 'gpu')
        tkwargs = {
            'batch_size': 1,  # does not matter
            'num_workers': self.cfgs.worker if not data_on_gpu else 0,
            'pin_memory': not data_on_gpu,
            'drop_last': True
        }
        loader, sampler = self.set_dataset('train', tkwargs)
        if sampler is not None:  # in case sample can not be distributed equally among machine, do not want miss
            self.reset_sampler(sampler, epoch)

        # concat all batch. Just do it once in beginning.
        data = self.concat_train_batch(loader)

        return data

    def concat_train_batch(self, loader):
        """Concat all elements from different samples together.
        The concat tensor should be in (N_img, HW, ...) shape
        """
        self.logger.add_log('Concat all training rays...')

        concat_data = {}
        total_item = len(loader)

        # append all dict from different image
        tensor_shape = {}
        dtype = None
        device = None
        for data in loader:
            for key in POTENTIAL_KEYS:
                if key in data:
                    tensor_shape[key] = data[key].shape
                    dtype = data[key].dtype
                    device = data[key].device
            concat_data['H'] = int(data['H'][0])
            concat_data['W'] = int(data['W'][0])
            break

        for key, t_shape in tensor_shape.items():
            concat_data[key] = torch.empty((total_item, *t_shape[1:]), dtype=dtype, device=device)

        for i, data in enumerate(loader):
            for key in POTENTIAL_KEYS:
                if key in data:
                    concat_data[key][i:i + 1] = data[key]

        return concat_data

    def process_train_data(self):
        """Process all concat data"""
        self.data['train'] = self.train_pipeline.process_train_data(self.logger, self.data['train'])

    def get_train_batch(self):
        """Get the train batch base on mode"""
        data_batch = self.train_pipeline.get_train_batch(self.data['train'])

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
        loader = torch.utils.data.DataLoader(
            dataset, sampler=sampler, shuffle=(sampler is None and mode != 'eval'), **tkwargs
        )

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
    def train_step_writer(
        self, epoch, step, step_in_epoch, loss, learning_rate, global_step, feed_in, inputs, output, **kwargs
    ):
        """Write to monitor for saving, add params"""
        super().train_step_writer(
            epoch, step, step_in_epoch, loss, learning_rate, global_step, feed_in, inputs, output, **kwargs
        )

        # write psnr
        if self.train_metric is not None:
            if epoch % self.cfgs.progress.epoch_loss == 0 and step % self.cfgs.progress.iter_loss == 0:
                metric_msg = 'Epoch {:06d} - Iter {}/{} - lr {:.8f}: '.format(
                    epoch, step, step_in_epoch - 1, learning_rate
                )
                metric = float(self.train_metric['metric'](feed_in, output))
                metric_msg += '{} [{:.3f}] '.format(self.train_metric['name'], metric)
                self.logger.add_log(metric_msg)

                self.monitor.add_scalar(self.train_metric['name'], metric, global_step)

        # write params
        if 'params' in output:
            for k, v in output['params'][0].items():
                self.monitor.add_scalar(k, v, global_step, mode='train')

    @master_only
    def print_loss_msg(self, epoch, step, step_in_epoch, loss, learning_rate):
        """Get the loss msg info and show in logger"""
        loss_msg = 'Epoch {:06d} - Iter {}/{} - lr {:.8f}: '.format(epoch, step, step_in_epoch - 1, learning_rate)
        for key in loss['names']:
            loss_msg += '{} [{:.4f}] | '.format(key, float(loss[key]))
        loss_msg += '--> Loss Sum [{:.4f}]'.format(float(loss['sum']))
        self.logger.add_log(loss_msg)

    def step_optimize(self, epoch, step, feed_in):
        """Set get progress for training"""
        output = self.model(feed_in, get_progress=self.get_progress, cur_epoch=epoch, total_epoch=self.total_epoch)
        loss = self.calculate_loss(feed_in, output)

        self.optimizer.zero_grad()
        loss['sum'].backward()

        # grad clipping and step
        self.clip_gradients(epoch)
        self.optimizer.step()

        # ema update
        if self.ema is not None:
            self.ema.ema_step()

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

                # psnr
                val_metric = None
                if self.train_metric is not None:
                    val_metric = float(self.train_metric['metric'](inputs, output))

                break  # only one batch per val epoch

            if count == 0:
                self.logger.add_log('Not batch was sent to valid...')
                self.model.train()
                return

            # get epoch average
            loss_summary.cal_average()

            if loss_summary.get_avg_summary() is not None:
                self.monitor.add_loss(loss_summary.get_avg_summary(), global_step, mode='val')
                val_msg = 'Validation Avg Loss --> Sum [{:.3f}]'.format(loss_summary.get_avg_sum())

                if val_metric is not None:
                    self.monitor.add_scalar(self.train_metric['name'], val_metric, global_step, mode='val')
                    val_msg += ' | {} --> [{:.3f}] '.format(self.train_metric['name'], val_metric)

                self.logger.add_log(val_msg)

        # release gpu memory
        self.empty_gpu_cache()

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
        self.data['inference'].run_infer(self.model, self.get_model_feed_in, eval_dir_epoch)

        # release gpu memory
        self.empty_gpu_cache()

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
        write_progress_imgs(files, folder, self.model, epoch, step, global_step, eval, self.radius, self.volume_dict)

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

        # optimize the model for its bounding structure
        self.model.optimize(epoch)
        if self.model.get_fg_model().get_obj_bound_type() == 'volume' and \
                epoch % self.cfgs.progress.epoch_loss == 0:
            volume = self.model.get_fg_model().get_obj_bound_structure()
            if volume.get_voxel_bitfield() is not None:
                occ_ratio = volume.get_n_occupied_voxel() / volume.get_n_voxel()
                self.logger.add_log('Remaining voxel ratio is {:.2f}%'.format(occ_ratio * 100.0))

        # remake train dataset train crop
        crop_shuffle = self.train_pipeline.check_crop_shuffle(epoch)
        full_shuffle = self.train_pipeline.check_full_shuffle()
        if crop_shuffle:
            self.data['train'] = self.set_train_dataset()
            self.process_train_data()
        else:
            if full_shuffle:
                if not self.train_pipeline.get_info('sample_cross_view'):  # reset data in this mode
                    self.data['train'] = self.set_train_dataset()
                self.process_train_data()

        # each epoch just contains one step for nerf training
        t_start = time.time()
        loss_all = self.train_step(epoch, 0, step_in_epoch, self.get_train_batch())
        epoch_time = time.time() - t_start
        if epoch % self.cfgs.progress.epoch_loss == 0:
            self.logger.add_log('Epoch time {:.3f} s/iter'.format(epoch_time))

        self.lr_scheduler.step()

        self.empty_gpu_cache()

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

        # init data handling
        self.process_train_data()

        loss_all = 0.0
        for epoch in range(self.cfgs.progress.start_epoch, self.cfgs.progress.epoch):
            # train and record speed
            self.model.train()

            loss_all, step_in_epoch = self.train_epoch(epoch)

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
