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
    """Creates the optimizer. Now support [adam, sgd, lbfgs, rmsprop]. """
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
