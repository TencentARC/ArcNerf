# -*- coding: utf-8 -*-

import torch.optim as optim


def create_optimizer(
    parameters,
    optim_type='adam',
    lr=1e-3,
    momentum=0.9,
    use_nesterov=True,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.0,
    centered=False,
    rmsprop_alpha=0.99,
    maxiters=20,
    **kwargs
):
    """Creates the optimizer.

     Args:
         parameters: parameters for training
         optim_type:  Now support [adam, sgd, lbfgs, rmsprop]. By default adam.
         lr: init learning rate. By default 1e-3.
         momentum: momentum. By default 0.9.
         use_nesterov: nesterov in sgd
         beta1: used for adam. By default 0.9
         beta2: used for adam. By default 0.999
         eps: eps in denominator. By default 1e-8
         weight_decay: weight decay for parameter regularization. By default 0.0
         centered: centered rmsprop
         rmsprop_alpha: alpha in rmsprop
         maxiters: maxiters for lbfgs
     """
    if optim_type == 'adam':
        return optim.Adam(parameters, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    elif optim_type == 'sgd':
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=use_nesterov)
    elif optim_type == 'lbfgs':
        return optim.LBFGS(parameters, lr=lr, max_iter=maxiters)
    elif optim_type == 'rmsprop':
        return optim.RMSprop(
            parameters,
            lr=lr,
            epsilon=eps,
            alpha=rmsprop_alpha,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered
        )
    else:
        raise ValueError('Optimizer {} not supported!'.format(optim_type))
