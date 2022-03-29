# -*- coding: utf-8 -*-

import numpy as np
import torch


def np_wrapper(func, *args):
    """ Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    Reference from VideoPose3d: https://github.com/facebookresearch/VideoPose3D/blob/master/common/utils.py
    """
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        return result.numpy()
    else:
        return result
