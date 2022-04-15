# -*- coding: utf-8 -*-

import numpy as np
import torch

from arcnerf.datasets import get_dataset
from arcnerf.datasets.transform.augmentation import get_transforms
from arcnerf.eval.eval_func import run_eval
from arcnerf.loss import build_loss
from arcnerf.metric import build_metric
from arcnerf.models import build_model
from common.loss.loss_dict import LossDictCounter
from common.trainer.basic_trainer import BasicTrainer
from common.utils.cfgs_utils import valid_key_in_cfgs, get_value_from_cfgs_field
from common.utils.img_utils import img_to_uint8
from common.utils.torch_utils import torch_to_np
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
                output = self.model(feed_in, get_progress=True)
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
        if inputs['H'][0] * inputs['W'][0] != inputs['img'].shape[1]:  # sampled rays, do not show anything
            return None

        names = []
        images = []
        idx = 0  # only sample from the first
        w, h = int(inputs['W'][idx]), int(inputs['H'][idx])
        # origin image, mask
        img = img_to_uint8(torch_to_np(inputs['img'][idx]).reshape(h, w, 3))  # (H, W, 3)
        mask = torch_to_np(inputs['mask'][idx]).reshape(h, w) if 'mask' in inputs else None  # (H, W)
        mask = (255.0 * mask).astype(np.uint8)[..., None].repeat(3, axis=-1) if mask is not None else None  # (H, W, 3)

        # pred rgb + img + error
        pred_rgb = ['rgb_coarse', 'rgb_fine', 'rgb']
        for pred_name in pred_rgb:
            if pred_name in output:
                pred_img = img_to_uint8(torch_to_np(output[pred_name][idx]).reshape(h, w, 3))  # (H, W, 3)
                error_map = np.abs(img - pred_img)  # (H, W, 3)
                pred_cat = np.concatenate([img, pred_img, error_map], axis=1)  # (H, 3W, 3)
                names.append(pred_name)
                images.append(pred_cat)
        # depth, norm and put to uint8(0-255)
        pred_depth = ['depth_coarse', 'depth_fine', 'depth']
        for pred_name in pred_depth:
            if pred_name in output:
                pred_depth = torch_to_np(output[pred_name][idx]).reshape(h, w)  # (H, W), 0~1
                pred_depth = (255.0 * pred_depth / (pred_depth.max() + 1e-8)).astype(np.uint8)
                pred_cat = np.concatenate([img, pred_depth[..., None].repeat(3, axis=-1)], axis=1)  # (H, 2W, 3)
                names.append(pred_name)
                images.append(pred_cat)
        # mask
        pred_mask = ['mask_coarse', 'mask_fine', 'mask']
        for pred_name in pred_mask:
            if pred_name in output:
                pred_mask = torch_to_np(output[pred_name][idx]).reshape(h, w)  # (H, W), 0~1, obj area with 1
                pred_mask = (255.0 * pred_mask).astype(np.uint8)[..., None].repeat(3, axis=-1)  # (H, W, 3), 0~255
                mask_img = (255 - pred_mask) + img  # (H, W, 3), white bkg
                if mask is not None:
                    error_map = (255.0 * np.abs(pred_mask - mask)).astype(np.uint8)  # (H, W, 3), 0~255
                    pred_cat = np.concatenate([mask_img, pred_mask, error_map], axis=1)  # (H, 3W, 3)
                else:
                    pred_cat = np.concatenate([mask_img, pred_mask], axis=1)  # (H, 2W, 3)
                names.append(pred_name)
                images.append(pred_cat)

        dic = {'names': names, 'imgs': images}

        return dic

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
