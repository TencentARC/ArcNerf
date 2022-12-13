# -*- coding: utf-8 -*-

import functools
import random

import numpy as np
import torch
import torch.distributed as dist


def set_random_seed(seed):
    """Set a random seed for training """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_dist_info():
    """Get dist rank and world size from torch.dist"""
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):
    """Wrapper that only run on rank=0 node"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def calc_max_grad(cfgs, parameters, to_cuda=True):
    """Calculate max grad for training supervision.
       An important remind is that you should do it on all ranks, otherwise all_reduce gets wrong.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if len(parameters) == 0:  # no valid params
        return 0.0

    max_norm = max(p.grad.data.abs().max() for p in parameters)
    max_norm_reduced = torch.FloatTensor([max_norm])
    if to_cuda:
        max_norm_reduced = max_norm_reduced.cuda()

    if cfgs.dist.world_size > 1:  # Every time you call all_reduce, every node should do it. Otherwise deadlock.
        torch.distributed.all_reduce(max_norm_reduced, op=torch.distributed.ReduceOp.MAX)
    return max_norm_reduced[0].item()
