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


def torch_to_np(tensor: torch.Tensor):
    """Torch tensor to numpy array"""
    return tensor.detach().cpu().numpy()


def torch_from_np_with_ref(np_array: np.ndarray, tensor_ref: torch.Tensor):
    """Create a torch.tensor on the same device of tensor_ref and with same dtype"""
    return torch.Tensor(np_array, dtype=tensor_ref.dtype, device=tensor_ref.device)


def set_tensor_to_zeros(tensor: torch.Tensor, a_tol=1e-5):
    """Set tensor with very small value as 0"""
    tensor[torch.abs(tensor) < a_tol] = 0.0

    return tensor


def chunk_processing(func, chunk_size, *args):
    """Processing array by func in chunk size, in case direct processing takes too much memory

    Args:
        func: the func processing array
        chunk_size: chunk size for each forward. <=0 means directly process.
        *args: containing arrays for input, assume each (B, d_in), torch or np array.
                If None, do not process it, The chunk division is on B-dim

    Returns:
        out: list of torch or np array in (B, d_out)
    """
    if chunk_size <= 0:
        return func(*args)

    # get batch size, you should assume all inputs have same shape in first dim.
    batch_size = 0
    for array in args:
        if array is not None:
            batch_size = array.shape[0]
    if batch_size == 0:
        return func(*args)

    # chunk processing
    out = []
    for i in range(0, batch_size, chunk_size):
        args_slice = get_batch_from_list(*args, start_idx=i, end_idx=i + chunk_size)
        chunk_out = func(*args_slice)
        if isinstance(chunk_out, tuple):
            chunk_out = list(chunk_out)
        elif isinstance(chunk_out, list):
            pass
        else:
            chunk_out = [chunk_out]
        out.append(chunk_out)

    # concat all field
    n_out_field = len(out[0])
    out_cat = out[0]
    for i in range(1, len(out)):
        for field_id in range(n_out_field):
            if out[i][field_id] is None:
                pass
            elif isinstance(out[i][field_id], np.ndarray):
                out_cat[field_id] = np.concatenate([out_cat[field_id], out[i][field_id]], axis=0)
            elif isinstance(out[i][field_id], torch.Tensor):
                out_cat[field_id] = torch.cat([out_cat[field_id], out[i][field_id]], dim=0)
            else:
                raise NotImplementedError('Invalid input type {}'.format(type(out[i][field_id])))

    if n_out_field == 1:
        return out_cat[0]
    else:
        return tuple(out_cat)


def get_batch_from_list(*args, start_idx, end_idx):
    """Get the batch of list from *args, each one can be an array or None

    Args:
        *args: containing arrays for input, assume each (B, d_in), torch or np array.
                If None, do not process it, The chunk division is on B-dim
        start_idx: start idx for selection.
        end_idx: end idx for selection. Valid for value > B

    Returns:
        out: list of torch or np array in (B, d_out)
    """
    return [a[start_idx:end_idx] if a is not None else None for a in args]


def mean_tensor_by_mask(tensor: torch.Tensor, mask: torch.Tensor, keep_batch=False):
    """Mean tensor with mask by batch size.
    Each tensor will be multiplied by mask and norm except first dim.

    Args:
        tensor: (B, anyshape), only need the first dim as batchsize, other are random
        mask: (B, anyshape), 0~1 with same shape as tensor
        keep_batch: If True, return (B, ), else return single mean value. By default False

    Returns:
        tensor_mean: (B, ) if keep batch; else (1, )
    """
    assert len(tensor.shape) > 1, 'At least two dim...'
    assert tensor.shape == mask.shape, 'Dim not match...'
    reduce_dim = tuple(range(1, len(tensor.shape)))

    tensor_mask = tensor * mask
    tensor_mask_sum = torch.sum(tensor_mask, dim=reduce_dim)  # (B, )
    mask_sum = torch.sum(mask, dim=reduce_dim)  # (B, )
    tensor_mean = tensor_mask_sum / mask_sum  # (B, )

    if not keep_batch:
        tensor_mean = tensor_mean.mean()

    return tensor_mean
