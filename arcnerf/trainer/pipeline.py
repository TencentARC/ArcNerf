# -*- coding: utf-8 -*-

import math

import torch

from common.utils.cfgs_utils import get_value_from_cfgs_field, valid_key_in_cfgs


class Pipeline(object):
    """This class is for handling the training rays"""

    def __init__(self):
        self.train_sample_info = {'sample_mode': 'full', 'sample_cross_view': True}
        self.crop_max_epoch = None
        self.init_precrop = False
        self.scheduler_cfg = None

    def setup_cfgs(self, cfgs):
        """Set up the cfgs for pipeline."""
        self.scheduler_cfg = cfgs

    def set_info(self, key, value):
        """set the train sample info"""
        self.train_sample_info[key] = value

    def get_info(self, key=None):
        """Get the info by key"""
        if key is None:
            return self.train_sample_info
        return self.train_sample_info[key]

    def set_n_rays(self, logger, n_rays):
        """Set up init num of rays"""
        self.set_info('n_rays', n_rays)
        logger.add_log('Num of rays for each training batch: {}'.format(n_rays))

    def check_crop_shuffle(self, epoch):
        """Check whether crop_epoch exit and get the inputs again"""
        crop_shuffle = self.crop_max_epoch is not None and epoch >= self.crop_max_epoch

        return crop_shuffle

    def check_full_shuffle(self):
        """Check whether all the data have been trained in full mode"""
        # shuffle data only after run up in full mode
        full_mode = (self.get_info('sample_mode') == 'full')
        full_run_up = (self.get_info('sample_total_count') >= self.get_info('total_samples'))
        full_shuffle = full_mode and full_run_up

        return full_shuffle

    def process_train_data(self, logger, train_data):
        """Process all concat data

        Args:
            logger: for logging
            train_data: a dict with all concat inputs
        """
        # reset sample index
        self.set_info('sample_img_count', 0)
        self.set_info('sample_total_count', 0)

        logger.add_log('-' * 60)
        logger.add_log('handle training samples...')

        # crop center image
        train_data = self.step_crop_center_image(logger, train_data)

        # ray sample preparation
        train_data = self.step_ray_sample(logger, train_data)

        # dynamic batch_size
        self.step_dynamic_bs(logger)

        # bkg color handling
        self.step_bkg_color(logger, train_data)

        # log information
        logger.add_log(
            'Need {} epoch to run all the {} rays...'.format(
                math.ceil(float(self.get_info('total_samples')) / float(self.get_info('n_rays'))),
                self.get_info('total_samples')
            )
        )

        logger.add_log('-' * 60)

        # set h/w as 0 to avoid writing of process image during training
        train_data['H'] = [0]
        train_data['W'] = [0]

        return train_data

    def step_crop_center_image(self, logger, train_data):
        """Crop the center images and get rays. You should not shuffle the init images."""
        if self.scheduler_cfg is not None and valid_key_in_cfgs(self.scheduler_cfg, 'precrop'):
            keep_ratio = get_value_from_cfgs_field(self.scheduler_cfg.precrop, 'ratio', 1.0)

            if keep_ratio < 1.0 and not self.init_precrop:
                self.init_precrop = True  # only precrop in the init stage
                self.crop_max_epoch = get_value_from_cfgs_field(self.scheduler_cfg.precrop, 'max_epoch', None)

                if self.crop_max_epoch is not None:
                    logger.add_log('Crop sample on first {} epoch'.format(self.crop_max_epoch))
                logger.add_log('Crop training samples...keep ratio - {}'.format(keep_ratio))

                h, w = train_data['H'], train_data['W']
                for k, v in train_data.items():
                    if isinstance(v, torch.Tensor):
                        full_tensor = v.view(-1, h, w, *v.shape[2:])  # (N, H, W, ...)
                        dh, dw = int((1 - keep_ratio) * h / 2.0), int((1 - keep_ratio) * w / 2.0)
                        crop_tensor = full_tensor[:, dh:-dh, dw:-dw, ...]  # (N, H_c, W_c, ...)
                        train_data['H'], train_data['W'] = crop_tensor.shape[1], crop_tensor.shape[2]
                        train_data[k] = crop_tensor.reshape(crop_tensor.shape[0], -1, *crop_tensor.shape[3:])
            else:
                self.crop_max_epoch = None
        else:
            self.crop_max_epoch = None

        # record down the new shape
        for k, v in train_data.items():
            if isinstance(v, torch.Tensor):
                total_sample = v.shape[0] * v.shape[1]
                self.set_info('total_samples', total_sample)
                self.set_info('n_train_img', v.shape[0])
                self.set_info('n_train_hw', v.shape[1])

        return train_data

    def step_ray_sample(self, logger, train_data):
        """Prepare all rays for sampling, but do not do the sample of batch"""
        # reset the mode
        if valid_key_in_cfgs(self.scheduler_cfg, 'ray_sample'):
            self.set_info('sample_mode', get_value_from_cfgs_field(self.scheduler_cfg.ray_sample, 'mode', 'full'))
            self.set_info(
                'sample_cross_view', get_value_from_cfgs_field(self.scheduler_cfg.ray_sample, 'cross_view', True)
            )

        assert self.get_info('sample_mode') in ['random', 'full'], \
            'Invalid mode {}'.format(self.get_info('sample_mode'))

        logger.add_log(
            'Sample mode: {}, Cross view: {}'.format(self.get_info('sample_mode'), self.get_info('sample_cross_view'))
        )

        # prepare all the rays in (1, nhw, dim)
        if self.get_info('sample_mode') == 'full':
            if self.get_info('sample_cross_view'):  # directly concat and shuffle all rays from all images
                random_idx = torch.randperm(self.get_info('total_samples'))

            else:
                # concat batches from different images in sequence. By last batches may mix different rays
                logger.add_log('Merge rays from different images into continuous batches..')
                n_train_hw = self.get_info('n_train_hw')
                n_rays = self.get_info('n_rays')
                random_idx_per_img = torch.randperm(n_train_hw)
                random_idx = []
                # get the index in concat images
                for start_idx in range(0, n_train_hw, n_rays):
                    random_img_idx = torch.randperm(self.get_info('n_train_img'))
                    for img_idx in random_img_idx:
                        random_idx.append(img_idx * n_train_hw + random_idx_per_img[start_idx:start_idx + n_rays])
                random_idx = torch.cat(random_idx, dim=0)

            # to (1, nhw, dim) and sample
            for k, v in train_data.items():
                if isinstance(v, torch.Tensor):
                    train_data[k] = v.view(1, -1, *v.shape[2:])[:, random_idx, ...]  # (1, n_total, ...)

        # random mode, keep the rays in (n, hw, dim)
        elif self.get_info('sample_mode') == 'random':
            pass

        return train_data

    def step_dynamic_bs(self, logger):
        """Adjust the dynamic batch size.
        Max_batch_size is set to forbid huge size of rays for extreme sparse view
        """

        if valid_key_in_cfgs(self.scheduler_cfg, 'dynamic_batch_size') \
                and get_value_from_cfgs_field(self.scheduler_cfg.dynamic_batch_size, 'update_epoch', 0) > 0:
            update_epoch = self.scheduler_cfg.dynamic_batch_size.update_epoch
            max_batch_size = get_value_from_cfgs_field(self.scheduler_cfg.dynamic_batch_size, 'max_batch_size', 32768)
            self.set_info('dynamic_batch_size', update_epoch)
            self.set_info('dynamic_max_batch_size', max_batch_size)
            logger.add_log(
                'Dynamically adjust training batch size every {} epoches...max bs allow {}'.format(
                    update_epoch, max_batch_size
                )
            )
            assert not (self.get_info('sample_mode') == 'full' and not self.get_info('sample_cross_view')), \
                'Not allow full image without cross view'
        else:
            self.set_info('dynamic_batch_size', 0)

    def step_bkg_color(self, logger, train_data):
        """bkg color handling. Just log information"""
        if self.scheduler_cfg is not None and valid_key_in_cfgs(self.scheduler_cfg, 'bkg_color') \
                and 'mask' in train_data.keys():
            logger.add_log('Train with bkg color: {}'.format(self.scheduler_cfg.bkg_color.color))

    def get_train_batch(self, train_data, epoch, model):
        """Get the training batch from prepared inputs"""
        data_batch = {}

        # update n_rays
        self.fetch_step_update_dynamic_bs(epoch, model)

        # fetch data by ray sample
        data_batch = self.fetch_step_ray_sample(train_data, data_batch)

        # handle bkg
        data_batch = self.fetch_step_bkg_color(data_batch)

        # other type data
        data_batch = self.fetch_step_other_type(train_data, data_batch)

        return data_batch

    def fetch_step_update_dynamic_bs(self, epoch, model):
        """Update dynamic batch size"""
        n_rays = self.get_info('n_rays')

        if self.get_info('dynamic_batch_size') > 0:
            update_epoch = self.get_info('dynamic_batch_size')
            if epoch % update_epoch == 0 and epoch > 500:  # fix warmup
                try:
                    dynamic_factor = model.get_dynamicbs_factor()
                except AttributeError:
                    dynamic_factor = model.module.get_dynamicbs_factor()

                def div_round_up(val, divisor):
                    return (val + divisor - 1) // divisor

                # in case too many rays together
                dynamic_n_rays = min(
                    int(div_round_up(n_rays * dynamic_factor, 128) * 128), self.get_info('dynamic_max_batch_size')
                )
                self.set_info('n_rays', dynamic_n_rays)

    def fetch_step_ray_sample(self, train_data, data_batch):
        """Fetch rays from full tensor by mode"""
        total_samples = self.get_info('total_samples')
        sample_total_count = self.get_info('sample_total_count')
        n_rays = self.get_info('n_rays')
        n_train_hw = self.get_info('n_train_hw')
        n_train_img = self.get_info('n_train_img')

        # random sample, do not need to keep count
        if self.get_info('sample_mode') == 'random':
            if self.get_info('sample_cross_view'):
                # when the ray is large, it takes too much time for indexing
                random_idx = torch.randperm(total_samples)[:n_rays]  # random from all rays
                for k, v in train_data.items():
                    if isinstance(v, torch.Tensor):  # tensor in (n_images, n_rays_per_image, ...)
                        data_batch[k] = v.view(1, -1, *v.shape[2:])[:, random_idx, ...]

            else:  # not cross view, sample in each image randomly
                randim_img_idx = torch.randint(0, n_train_img, [1])[0]
                random_idx = torch.randperm(n_train_hw)[:n_rays]  # random from single image rays
                for k, v in train_data.items():
                    if isinstance(v, torch.Tensor):  # tensor in (n_images, n_rays_per_image, ...)
                        data_batch[k] = v[randim_img_idx, random_idx, ...].unsqueeze(0)

        # full sample, keep count record
        elif self.get_info('sample_mode') == 'full':
            assert sample_total_count < total_samples, 'All rays have been sampled, please reset train dataset...'

            for k, v in train_data.items():
                if isinstance(v, torch.Tensor):  # tensor in (1, n_images * n_rays_per_image, ...)
                    data_batch[k] = v[:, sample_total_count:sample_total_count + n_rays, ...]

            self.set_info('sample_total_count', self.get_info('sample_total_count') + n_rays)

        return data_batch

    def fetch_step_bkg_color(self, data_batch):
        """Handle the bkg color. """
        if self.scheduler_cfg is not None and valid_key_in_cfgs(self.scheduler_cfg, 'bkg_color') and \
                'mask' in data_batch.keys():

            assert 'mask' in data_batch.keys(), 'You must have mask in inputs...'

            if get_value_from_cfgs_field(self.scheduler_cfg.bkg_color, 'color', 'random') == 'random':
                bkg_color = torch.rand_like(data_batch['img'], device=data_batch['img'].device).detach()
            else:
                bkg_color = torch.tensor(
                    self.scheduler_cfg.bkg_color.color, dtype=data_batch['img'].dtype, device=data_batch['img'].device
                ).detach()[None, None]  # (B, N, 3)

                bkg_color = torch.ones_like(data_batch['img'], device=data_batch['img'].device).detach() * bkg_color

            # rewrite bkg color and image for loss computation
            img = data_batch['img'] * data_batch['mask'][..., None] + (1.0 - data_batch['mask'][..., None]) * bkg_color
            data_batch['bkg_color'] = bkg_color
            data_batch['img'] = img

        return data_batch

    @staticmethod
    def fetch_step_other_type(train_data, data_batch):
        """Other type of data, just copy"""
        for k, v in train_data.items():
            if not isinstance(v, torch.Tensor):
                data_batch[k] = v

        return data_batch
