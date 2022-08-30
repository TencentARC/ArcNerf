# -*- coding: utf-8 -*-

import torch


class EMA(object):
    """A moving average optimization for params with grad
        Ref: https://github.com/Jittor/JNeRF/blob/master/python/jnerf/optims/ema.py
    """

    def __init__(self, model, decay):
        self.model = model
        self.old_avg = self.get_model_params()
        self.decay = decay
        self.n_step = 0

    def get_model_params(self):
        params = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                params[n] = p.clone()

        return params

    def set_n_step(self, n_step):
        """Set the step"""
        self.n_step = n_step

    def ema_step(self):
        """step forward to update the params"""
        self.n_step += 1
        ema_debias_old = 1 - self.decay**(self.n_step - 1)
        ema_debias_new = 1.0 / (1 - self.decay**self.n_step)

        # update param
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    assert n in self.old_avg.keys(), '{} not save in params...'.format(n)
                    old_avg = self.old_avg[n]
                    new_avg = ((1 - self.decay) * p + self.decay * old_avg * ema_debias_old) * ema_debias_new
                    p.copy_(new_avg.detach())
                    self.old_avg[n] = new_avg.detach()
