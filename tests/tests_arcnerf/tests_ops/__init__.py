# -*- coding: utf-8 -*-

import torch

from common.utils.torch_utils import get_start_time, get_end_time


def log_custom_benchmark(logger, func_name, torch_func, custom_fuc, inputs, n_iter=100):
    """Compare different method and log custom benchmark on cuda

    Args:
        logger: logger to write info
        func_name: str name of the func
        torch_func: torch implementation for comparsion. If None, only run on custom_func
        custom_fuc: custom ops in cuda.
        inputs: a list of input. For the tensors, they should be in cpu.
        n_iter: for average running time. By default 100. If n_iter <= 0, only runs once for output/grad, no timing.

    Returns:
        out_torch: list of torch output. Return None if torch_func is None
        out_custom: list of custom output
        out_custom_forward_only: list of custom output from forward-only mode
        grad_torch: list of torch grad. Return None if torch_func is None
        grad_custom: list of custom grad
    """
    if not torch.cuda.is_available():
        return None, None, None, None

    logger.add_log('_' * 60)
    logger.add_log(func_name)

    # to gpu. grad is on cpu tensor only
    inputs_gpu = []
    for input in inputs:
        if isinstance(input, torch.Tensor):
            inputs_gpu.append(input.cuda())
            logger.add_log('  Input dim: {}'.format(input.shape))
        else:
            inputs_gpu.append(input)

    logger.add_log('Time by iter {}'.format(n_iter))

    out_torch = None
    grad_torch = None
    t_forward_torch, t_backward_torch, t_forward_only_torch = None, None, None
    # torch implementation
    if torch_func is not None:
        # grad
        out_torch = torch_func(*inputs_gpu)
        loss = torch.sum((1.0 - out_torch)**2)
        loss.backward()

        # log the grad
        grad_torch = []
        for input in inputs:
            if isinstance(input, torch.Tensor):
                grad_torch.append(input.grad.clone())
            else:
                grad_torch.append(None)

        # grad from model
        if isinstance(torch_func, torch.nn.Module):
            for n, p in torch_func.named_parameters():
                if p.grad is not None:
                    grad_torch.append(p.grad.clone())

        # timing
        t_forward_torch = 0.0
        t_backward_torch = 0.0
        t_forward_only_torch = 0.0
        for _ in range(n_iter):
            # zeros the grad
            for input in inputs:
                input.grad.zero_()

            t0 = get_start_time()
            out_torch = torch_func(*inputs_gpu)
            t_forward_torch += get_end_time(t0)

            loss = torch.sum((1.0 - out_torch)**2)

            t0 = get_start_time()
            loss.backward()
            t_backward_torch += get_end_time(t0)

            # forward only
            with torch.no_grad():
                t0 = get_start_time()
                _ = torch_func(*inputs_gpu)
                t_forward_only_torch += get_end_time(t0)

        # log time
        if n_iter > 0:
            t_forward_torch = t_forward_torch / float(n_iter)
            logger.add_log('Torch Forward time {:.6f}s'.format(t_forward_torch))
            t_backward_torch = t_backward_torch / float(n_iter)
            logger.add_log('Torch Backward time {:.6f}s'.format(t_backward_torch))
            t_forward_only_torch = t_forward_only_torch / float(n_iter)
            logger.add_log('Torch Forward-Only time {:.6f}s'.format(t_forward_only_torch))

        # zeros the grad
        for input in inputs:
            input.grad.zero_()

    # custom cuda implementation
    # log the grad
    out_custom = custom_fuc(*inputs_gpu)
    loss = torch.sum((1.0 - out_custom)**2)
    loss.backward()

    grad_custom = []
    for input in inputs:
        if isinstance(input, torch.Tensor):
            grad_custom.append(input.grad.clone())
        else:
            grad_custom.append(None)

    with torch.no_grad():
        out_custom_forward_only = custom_fuc(*inputs_gpu)

    # grad from model
    if isinstance(custom_fuc, torch.nn.Module):
        for n, p in custom_fuc.named_parameters():
            if p.grad is not None:
                grad_custom.append(p.grad.clone())

    t_forward_custom = 0.0
    t_backward_custom = 0.0
    t_forward_only_custom = 0.0
    for _ in range(n_iter):
        # zeros the grad
        for input in inputs:
            if input.grad is not None:
                input.grad.zero_()

        t0 = get_start_time()
        out_custom = custom_fuc(*inputs_gpu)
        t_forward_custom += get_end_time(t0)

        loss = torch.sum((1.0 - out_custom)**2)

        t0 = get_start_time()
        loss.backward()
        t_backward_custom += get_end_time(t0)

        # forward only
        with torch.no_grad():
            t0 = get_start_time()
            _ = custom_fuc(*inputs_gpu)
            t_forward_only_custom += get_end_time(t0)

    # log time
    if n_iter > 0:
        t_forward_custom = t_forward_custom / float(n_iter)
        if t_forward_torch is None:
            logger.add_log('Custom Forward time {:.6f}s'.format(t_forward_custom))
        else:
            logger.add_log(
                'Custom Forward time {:.6f}s. Boost x{:.2f}'.format(
                    t_forward_custom, t_forward_torch / t_forward_custom
                )
            )

        t_backward_custom = t_backward_custom / float(n_iter)
        # log backward
        if t_backward_torch is None:
            logger.add_log('Custom Backward time {:.6f}s'.format(t_backward_custom))
        else:
            logger.add_log(
                'Custom Backward time {:.6f}s. Boost x{:.2f}'.format(
                    t_backward_custom, t_backward_torch / t_backward_custom
                )
            )

        t_forward_only_custom = t_forward_only_custom / float(n_iter)
        # log forward-only
        if t_forward_only_torch is None:
            logger.add_log('Custom Forward-Only time {:.6f}s'.format(t_forward_only_custom))
        else:
            logger.add_log(
                'Custom Forward-Only time {:.6f}s. Boost x{:.2f}'.format(
                    t_forward_only_custom, t_forward_only_torch / t_forward_only_custom
                )
            )

    logger.add_log('_' * 60)
    logger.add_log('\n')

    return out_torch, out_custom, out_custom_forward_only, grad_torch, grad_custom
