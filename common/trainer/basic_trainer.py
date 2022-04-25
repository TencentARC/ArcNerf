# -*- coding: utf-8 -*-

import os
import os.path as osp
import subprocess
import sys
import time

import torch

from common.loss.loss_dict import LossDictCounter
from common.metric.metric_dict import MetricDictCounter
from common.trainer.lr_scheduler import get_learning_rate_scheduler
from common.trainer.optimizer import create_optimizer
from common.utils.cfgs_utils import create_train_sh, dump_configs
from common.utils.logger import Logger
from common.utils.model_io import load_model, save_model
from common.utils.monitor import Monitor
from common.utils.train_utils import (calc_max_grad, master_only, set_random_seed)


class BasicTrainer(object):
    """Basic Trainer class. Use for all general deep learning project"""

    def __init__(self, cfgs):
        self.cfgs = cfgs

        # distributed params
        self.cfgs.dist.rank, self.cfgs.dist.local_rank, self.cfgs.dist.world_size = self.set_dist_rank()

        # set gpu for each process and init distribution
        self.gpu_id = self.set_gpu_id()
        self.device = 'cpu'
        if torch.cuda.is_available() and self.gpu_id >= 0:
            torch.cuda.set_device(self.gpu_id)
            self.device = 'gpu'

        if self.gpu_id >= 0 and self.cfgs.dist.world_size > 1:
            self.master_address = os.environ['MASTER_ADDR']
            self.master_port = os.environ['MASTER_PORT'] if 'MASTER_PORT' in os.environ else '11623'

        self.set_distributed()

        # set up exp folder
        self.set_dir()
        self.backup()

        self.logger = self.set_logger()
        self.set_reproducibility()

        # model
        model = self.get_model()
        self.model, self.optimizer = self.set_model_and_optimizer(model)

        # lr scheduler
        self.lr_scheduler = self.set_lr_scheduler()

        # data
        self.data = self.prepare_data()

        # loss
        self.loss_factory = self.set_loss_factory()
        self.monitor = self.set_monitor()

        # metric
        self.eval_metric = self.set_eval_metric()

        # set start_time
        self.time0 = time.time()

    def set_dist_rank(self):
        """Set up distributed rank and local rank"""
        if self.cfgs.dist.slurm:
            rank = int(os.environ['SLURM_PROCID'])
            local_rank = rank % torch.cuda.device_count()
            world_size = int(os.environ['SLURM_NTASKS'])
            node_list = os.environ['SLURM_NODELIST']
            addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
            os.environ['MASTER_ADDR'] = addr
        else:
            rank = int(os.getenv('RANK')) if os.getenv('RANK') is not None else 0
            local_rank = int(os.getenv('LOCAL_RANK')) if os.getenv('LOCAL_RANK') is not None else 0
            world_size = int(os.getenv('WORLD_SIZE')) if os.getenv('WORLD_SIZE') is not None else 1

        return rank, local_rank, world_size

    def set_gpu_id(self):
        """Set the gpu id for this local process. -1 will be cpu"""
        if isinstance(self.cfgs.gpu_ids, int) and self.cfgs.gpu_ids < 0:
            return -1

        if isinstance(self.cfgs.gpu_ids, int):
            self.cfgs.gpu_ids = [self.cfgs.gpu_ids]

        # device is set based on local rank
        gpu_id = self.cfgs.gpu_ids[self.cfgs.dist.local_rank % len(self.cfgs.gpu_ids)]

        return gpu_id

    def set_distributed(self):
        """Set up distributed for multiple process"""
        if self.cfgs.dist.world_size > 1:
            init_method = 'tcp://{}:{}'.format(self.master_address, self.master_port)
            torch.distributed.init_process_group(
                backend=self.cfgs.dist.distributed_backend,
                world_size=self.cfgs.dist.world_size,
                rank=self.cfgs.dist.rank,
                init_method=init_method
            )

    @master_only
    def set_dir(self):
        """Set up all the directory including events, checkpoints, eval, etc"""
        if self.cfgs.dir.data_dir is None:
            self.cfgs.dir.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
        if self.cfgs.dir.expr_dir is None:
            self.cfgs.dir.expr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../experiments'))
        self.cfgs.dir.expr_spec_dir = osp.join(self.cfgs.dir.expr_dir, getattr(self.cfgs, 'name', 'default'))
        self.cfgs.dir.checkpoint_dir = osp.join(self.cfgs.dir.expr_spec_dir, 'checkpoints')
        self.cfgs.dir.eval_dir = osp.join(self.cfgs.dir.expr_spec_dir, 'eval')
        self.cfgs.dir.event_dir = osp.join(self.cfgs.dir.expr_spec_dir, 'event')

        os.makedirs(self.cfgs.dir.expr_dir, exist_ok=True)
        os.makedirs(self.cfgs.dir.expr_spec_dir, exist_ok=True)
        os.makedirs(self.cfgs.dir.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cfgs.dir.eval_dir, exist_ok=True)
        os.makedirs(self.cfgs.dir.event_dir, exist_ok=True)

    @master_only
    def backup(self):
        """Backup experiment configs in result folder"""
        backup_config_path = osp.join(self.cfgs.dir.expr_spec_dir, 'cfg_backup.yaml')
        dump_configs(self.cfgs, backup_config_path)
        create_train_sh(
            self.cfgs.name + '_replicate', backup_config_path, 'train.py', save_dir=self.cfgs.dir.expr_spec_dir
        )

    def set_logger(self):
        """Set the train logger. Rank=0 node will write to train.log file"""
        if self.cfgs.dist.rank == 0:
            log_path = osp.join(self.cfgs.dir.expr_spec_dir, 'train.log')
            logger = Logger(rank=self.cfgs.dist.rank, path=log_path)
        else:
            logger = Logger(rank=self.cfgs.dist.rank)

        logger.add_log('Start Training! Expr Name "{}"'.format(self.cfgs.name))
        logger.add_log('Total Number of Nodes: {}.'.format(self.cfgs.dist.world_size))

        return logger

    def set_reproducibility(self):
        """Set the random seed given cfgs.random_seed. Helps to keep reproducibility"""
        if self.cfgs.dist.random_seed:
            self.logger.add_log('Require reproducibility, random seed as [{}].'.format(self.cfgs.random_seed))
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            set_random_seed(self.cfgs.random_seed)
        else:
            self.logger.add_log('Do not Require reproducibility.')
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    def set_model_and_optimizer(self, model):
        """Set up the model and optimizer and distribute to machines"""
        model = self.distribute_model(model)

        params = self.get_lr_params_groups(model)

        optimizer = create_optimizer(parameters=params, **self.cfgs.optim.__dict__)

        if self.cfgs.resume:
            if osp.isdir(self.cfgs.resume):  # will pick latest.pth as model
                model_path = osp.join(self.cfgs.resume, 'latest.pt.tar')
                if osp.isfile(model_path):
                    model = load_model(self.logger, model, optimizer, model_path, self.cfgs)
                else:
                    self.logger.add_log('No latest checkpoint found in dir {}'.format(self.cfgs.resume), level='error')
                    exit(1)
            elif osp.isfile(self.cfgs.resume):
                model = load_model(self.logger, model, optimizer, self.cfgs.resume, self.cfgs)
            elif self.cfgs.resume:
                self.logger.add_log('No checkpoint found at {}'.format(self.cfgs.resume), level='error')
                exit(1)
        else:
            self.logger.add_log('Random initialization. No checkpoint provided. Starts from epoch 0.')
            self.cfgs.progress.start_epoch = 0

        return model, optimizer

    def distribute_model(self, model):
        """Distribute the model on different machines"""
        if self.gpu_id < 0:
            return model

        if self.cfgs.dist.world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
        else:
            model.cuda()

        return model

    @staticmethod
    def get_lr_params_groups(model):
        """Get the params groups for different lr if required. Needed to be implemented in custom trainer"""
        params = model.parameters()

        return params

    def set_lr_scheduler(self):
        """Set the lr scheduler"""
        lr_scheduler = get_learning_rate_scheduler(
            self.optimizer,
            last_epoch=self.cfgs.progress.start_epoch - 1,
            total_epoch=self.cfgs.progress.epoch,
            **self.cfgs.optim.lr_scheduler.__dict__
        )
        self.logger.add_log('-' * 60)
        self.logger.add_log(
            'Optim Type - {}. Schedule Type - {}. Start lr - {}.'.format(
                self.cfgs.optim.optim_type, self.cfgs.optim.lr_scheduler.type, self.cfgs.optim.lr
            )
        )

        if self.cfgs.optim.clip_gradients > 0.0:
            self.logger.add_log('Clipping Gradient for progress set to {}.'.format(self.cfgs.optim.clip_gradients))

        if self.cfgs.optim.clip_gradients_warmup > 0.0 and self.cfgs.optim.clip_warmup >= 0:
            clip_warmup = self.cfgs.optim.clip_gradients_warmup
            clip_warmup_epoch = self.cfgs.optim.clip_warmup
            self.logger.add_log(
                'Clipping Gradient after {} Warm-up Epochs set to {}.'.format(clip_warmup_epoch, clip_warmup)
            )

        return lr_scheduler

    def get_grad_clip(self, epoch):
        """Get the actual grad for clipping by epoch """
        grad_clip = -1.0
        if 0 <= self.cfgs.optim.clip_warmup <= epoch:
            if self.cfgs.optim.clip_gradients_warmup > 0.0:
                grad_clip = self.cfgs.optim.clip_gradients_warmup

        else:
            if self.cfgs.optim.clip_gradients > 0.0:
                grad_clip = self.cfgs.optim.clip_gradients

        return grad_clip

    def clip_gradients(self, epoch):
        """grad clipping. Only clip when grad > 0.0"""
        grad_clip = self.get_grad_clip(epoch)
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), grad_clip)

    def debug_print_grad(self, epoch, step):
        """Print the max grad, and param grad information for debug. Only in debug mode"""
        if self.cfgs.debug.debug_mode and step % self.cfgs.progress.iter_loss == 0:
            grad_clip = self.get_grad_clip(epoch)
            if grad_clip > 0.0:
                self.logger.add_log('Grad clip set to {:.2f}'.format(grad_clip), level='debug')

            max_grad = calc_max_grad(self.cfgs, self.model.parameters(), to_cuda=(self.gpu_id >= 0))
            self.logger.add_log('Max Gradient abs value {:.2f}.'.format(max_grad), level='debug')

            if self.cfgs.debug.print_all_grad:
                for name, parms in self.model.named_parameters():
                    self.logger.add_log(
                        'name [{}] - grad_requires [{:.2f}] - weight [{:.2f}] - abs grad [{:.2f}].'.format(
                            name, parms.requires_grad, torch.mean(parms.data), torch.mean(parms.grad.data.abs())
                        ),
                        level='debug'
                    )

    @staticmethod
    def reset_sampler(sampler, epoch):
        """Reset sampler. It could be list single one"""
        if isinstance(sampler, list):
            for s in sampler:
                s.set_epoch(epoch)
        else:
            sampler.set_epoch(epoch)

    @master_only
    def set_monitor(self):
        """Set up monitor for all events"""
        monitor = Monitor(log_dir=self.cfgs.dir.event_dir)

        return monitor

    @master_only
    def save_model(self, epoch, loss, spec_name=None):
        """Save model. Only master node does it. """
        save_model(
            self.logger, self.model, self.optimizer, epoch, loss, self.cfgs.dir.checkpoint_dir, self.cfgs, spec_name
        )

    def calculate_loss(self, inputs, output):
        """Calculate the loss for inputs and output.
         You have to put the input field to the same device of output
        """
        loss = self.loss_factory(inputs, output)

        return loss

    @master_only
    def print_loss_msg(self, epoch, step, step_in_epoch, loss, learning_rate):
        """Get the loss msg info and show in logger"""
        loss_msg = 'Epoch {:06d} - Iter {}/{} - lr {:.8f}: '.format(epoch, step, step_in_epoch - 1, learning_rate)
        for key in loss['names']:
            loss_msg += '{} [{:.3f}] | '.format(key, float(loss[key]))
        loss_msg += '--> Loss Sum [{:.3f}]'.format(float(loss['sum']))
        self.logger.add_log(loss_msg)

    @master_only
    def save_progress(self, epoch, step, global_step, inputs, output, mode='train'):
        """Save progress img for tracking. For both training and val. By default write to monitor.
            You are allowed to send a list of imgs to write for each iteration.
        """
        files = self.render_progress_imgs(inputs, output)
        if files is None:
            return

        # only add the images from file. Other like figs, etc are not support for monitor. You need to overwrite it.
        if 'imgs' in files:
            for name, img in zip(files['imgs']['names'], files['imgs']['imgs']):
                self.monitor.add_img(name, img, global_step, mode=mode)

        if self.cfgs.progress.local_progress:
            progress_dir = osp.join(self.cfgs.dir.expr_spec_dir, 'progress')
            os.makedirs(progress_dir, exist_ok=True)
            progress_mode_dir = osp.join(progress_dir, mode)
            os.makedirs(progress_mode_dir, exist_ok=True)
            self.write_progress_imgs([files], progress_mode_dir, epoch, step, global_step, eval=False)

    @master_only
    def train_step_writer(self, epoch, step, step_in_epoch, loss, learning_rate, global_step, inputs, output, **kwargs):
        """Write to monitor for saving"""
        self.monitor.add_loss(loss, global_step, mode='train')
        self.monitor.add_scalar('learning_rate', learning_rate, global_step)

        # save loss logs
        if epoch % self.cfgs.progress.epoch_loss == 0 and step % self.cfgs.progress.iter_loss == 0:
            self.print_loss_msg(epoch, step, step_in_epoch, loss, learning_rate)

        # save progress results in training progress. Only do in certain epoch and iter
        if self.cfgs.progress.save_progress:
            if epoch % self.cfgs.progress.epoch_save_progress == 0:
                if step % self.cfgs.progress.iter_save_progress == 0:
                    self.save_progress(epoch, step, global_step, inputs, output, mode='train')

    def step_optimize(self, epoch, step, feed_in, inputs):
        """Core step function for optimize in one step. You can rewrite it if you need some more flexibility """
        output = self.model(feed_in)
        loss = self.calculate_loss(inputs, output)

        self.optimizer.zero_grad()
        loss['sum'].backward()

        # grad clipping and step
        self.clip_gradients(epoch)
        self.optimizer.step()

        # print grad for debug
        self.debug_print_grad(epoch, step)

        return output, loss

    def train_step(self, epoch, step, step_in_epoch, inputs):
        """Train for one step"""
        loss_all = 0.0
        try:
            feed_in, _ = self.get_model_feed_in(inputs, self.device)
            output, loss = self.step_optimize(epoch, step, feed_in, inputs)
            loss_all = loss['sum']

            # for simplicity, we don't broadcast the loss from all device.
            learning_rate = self.lr_scheduler.get_last_lr()[0]
            global_step = step_in_epoch * epoch + step

            # write to monitor, include loss, output/gt visual
            self.train_step_writer(epoch, step, step_in_epoch, loss, learning_rate, global_step, inputs, output)

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

    def train_epoch(self, epoch):
        """Train for one epoch. Each epoch return the final sum of loss and total num of iter in epoch"""
        self.logger.add_log('Epoch {:06d}:'.format(epoch))
        loss_all = 0.0
        step_in_epoch = len(self.data['train'])

        # refresh train sampler
        refresh = self.data['train_sampler'] is not None
        if refresh:
            self.reset_sampler(self.data['train_sampler'], epoch)

        for step, inputs in enumerate(self.data['train']):
            loss_all = self.train_step(epoch, step, step_in_epoch, inputs)

        self.lr_scheduler.step()

        return loss_all, step_in_epoch

    @master_only
    def eval_epoch(self, epoch):
        """Eval the epoch using test data"""
        if self.eval_metric is None:
            self.logger.add_log('No eval_metric provided... Check the setting...', level='warning')
            return

        self.logger.add_log('Eval on test data... Epoch {}'.format(epoch))
        eval_dir_epoch = osp.join(self.cfgs.dir.eval_dir, 'epoch_{:06d}'.format(epoch))
        os.makedirs(eval_dir_epoch, exist_ok=True)

        # eval and show
        self.model.eval()
        metric_summary = MetricDictCounter()
        metric_info, files = self.evaluate(
            self.data['eval'], self.model, metric_summary, self.device, self.cfgs.progress.max_samples_eval
        )
        if metric_info is None:
            self.logger.add_log('No evaluation perform...', level='warning')
        else:
            self.logger.add_log('Evaluation at Epoch {} Benchmark result. \n {}'.format(epoch, metric_info))
            if files is not None:
                self.write_progress_imgs(files, folder=eval_dir_epoch, eval=True)
                self.logger.add_log('Visual results add to {}'.format(eval_dir_epoch))

            eval_log_file = os.path.join(eval_dir_epoch, 'eval_log.txt')
            with open(eval_log_file, 'w') as f:
                f.writelines(metric_info)

        # release gpu memory
        if self.device == 'gpu':
            torch.cuda.empty_cache()

        self.model.train()

    @master_only
    def valid_epoch(self, epoch, step_in_epoch):
        """Validate the epoch.
           Remember to set eval mode at beginning and set train mode at the end
        """
        self.logger.add_log('Valid on data...')
        self.model.eval()
        loss_summary = LossDictCounter()
        count = 0
        global_step = (epoch + 1) * step_in_epoch
        for step, inputs in enumerate(self.data['val']):
            with torch.no_grad():
                feed_in, batch_size = self.get_model_feed_in(inputs, self.device)
                output = self.model(feed_in)
            if self.cfgs.progress.save_progress_val and step < self.cfgs.progress.max_samples_val:  # Just some samples
                self.save_progress(epoch, 0, global_step, inputs, output, mode='val')

            count += batch_size
            loss = self.calculate_loss(inputs, output)
            loss_summary(loss, batch_size)

        if count == 0:
            self.logger.add_log('No batch was sent to valid...')
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

    def train(self):
        """Train for the whole progress"""
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
            iter_time = epoch_time / float(step_in_epoch)
            self.logger.add_log('Epoch time {:.2f} min...({:.3f}s/iter)'.format(epoch_time / 60.0, iter_time))

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

    def get_model(self):
        """Get the model end2end by yourself"""
        raise NotImplementedError('Please implement the detail get_model method in child class...')

    def get_model_feed_in(self, inputs, device):
        """Get the model_feed_in data from input, which will be directly send into the network
        You have to put them to the save device
        """
        raise NotImplementedError('Please implement the detail get_model_feed_in method in child class...')

    def prepare_data(self):
        """prepare data for train/valid/test"""
        raise NotImplementedError('Please implement the detail prepare_data method in child class...')

    def set_loss_factory(self):
        """Set loss factory for all loss combined"""
        raise NotImplementedError('Please implement the detail loss factory function method in child class...')

    @master_only
    def set_eval_metric(self):
        """Set up the eval metric"""
        raise NotImplementedError('Please implement the detail set_eval_metric method in child class...')

    def evaluate(self, data, model, metric_summary, device, max_samples_eval):
        """Actual eval function for the model. """
        raise NotImplementedError('Please implement the detail evaluate method in child class...')

    def render_progress_imgs(self, inputs, output):
        """Actual render for progress image with label. It is perform in each step with a batch.
         Return a dict with list of image and filename. filenames should be irrelevant to progress
         Image should be in bgr with shape(h,w,3), which can be directly writen by cv.imwrite().
         Return None will not write anything.
         """
        raise NotImplementedError('Please implement the detail render function in child class...')

    def write_progress_imgs(self, files, folder, epoch=None, step=None, global_step=None, eval=False):
        """Actual function to write the progress images"""
        raise NotImplementedError('Please implement the detail write progress function in child class...')
