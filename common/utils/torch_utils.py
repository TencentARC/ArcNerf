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


def torch_to_np(tensor):
    """Torch tensor to numpy array"""
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        raise NotImplementedError('Please use torch tensor or np array')


def torch_from_np_with_ref(np_array: np.ndarray, tensor_ref: torch.Tensor):
    """Create a torch.tensor on the same device of tensor_ref and with same dtype"""
    return torch.Tensor(np_array, dtype=tensor_ref.dtype, device=tensor_ref.device)


def set_tensor_to_zeros(tensor: torch.Tensor, a_tol=1e-5):
    """Set tensor with very small value as 0"""
    tensor[torch.abs(tensor) < a_tol] = 0.0

    return tensor


def is_torch_or_np(tensor):
    """Test whether a tensor is torch or np array"""
    return isinstance(tensor, torch.Tensor) or isinstance(tensor, np.ndarray)


def chunk_processing(func, chunk_size, gpu_on_func, *args):
    """Processing array by func in chunk size, in case direct processing takes too much memory
    Since the inputs could be large to put in gpu together and concat in gpu, you can set gpu_on_func=True,
    and just make the inputs in cpu, each chunk will bring tensor into gpu and back to cpu to avoid explosion.

    Args:
        func: the func processing array and other parameters
        chunk_size: chunk size for each forward. <=0 means directly process.
        gpu_on_func: If True, will move torch tensor to 'gpu' before func
                     and move back to 'cpu' after it(only when tensor original in cpu'), in case large gpu consumption.
        *args: containing arrays for input, assume each (B, d_in), torch or np array. The chunk division is on B-dim.
               Can be other params like float/str/bool/none, etc. But at least contains one array.
               If it is a dict, will slice each item and output the dict as well. (Do not use nested dict)

    Returns:
        out: list of torch or np array in (B, d_out), other values in list
             if dict is inside the output list and it contain array, the array should be concat as well
    """
    if chunk_size <= 0:
        return func(*args)

    # get batch size, you should assume all inputs have same shape in first dim.
    batch_size = 0
    for array in args:
        if is_torch_or_np(array):
            assert batch_size == 0 or batch_size == array.shape[0], 'Batch size for array not matched...'
            batch_size = array.shape[0]
        elif isinstance(array, dict):
            for v in array.values():
                if is_torch_or_np(v):
                    assert batch_size == 0 or batch_size == v.shape[0], 'Batch size for array not matched...'
                    batch_size = v.shape[0]
        else:
            pass

    if batch_size == 0:
        return func(*args)

    # chunk processing
    out = []
    for i in range(0, batch_size, chunk_size):
        args_slice, move_to_cpu = get_batch_from_list(
            *args, start_idx=i, end_idx=i + chunk_size, gpu_on_func=gpu_on_func
        )
        chunk_out = func(*args_slice)
        # combine results
        if isinstance(chunk_out, tuple):
            chunk_out = list(chunk_out)
        elif isinstance(chunk_out, list):
            pass
        else:
            chunk_out = [chunk_out]
        # clean unused gpu memory and move back result to cpu
        if move_to_cpu:
            for idx, o in enumerate(chunk_out):
                if isinstance(o, torch.Tensor):
                    chunk_out[idx] = o.cpu()
                elif isinstance(o, dict):
                    for k, v in o.items():
                        if isinstance(v, torch.Tensor):
                            chunk_out[idx][k] = v.cpu()

                torch.cuda.empty_cache()

        out.append(chunk_out)

    # concat all field, consider dict as well
    n_chunk = len(out)
    n_out_field = len(out[0])
    out_cat = out[0]
    for i, o in enumerate(out_cat):
        if not isinstance(o, dict) and not is_torch_or_np(o):
            out_cat[i] = [o]
        elif isinstance(o, dict):
            for k, v in o.items():
                if not is_torch_or_np(v):
                    out_cat[i][k] = [v]

    for i in range(1, n_chunk):
        for field_id in range(n_out_field):
            if isinstance(out[i][field_id], np.ndarray):
                out_cat[field_id] = np.concatenate([out_cat[field_id], out[i][field_id]], axis=0)
            elif isinstance(out[i][field_id], torch.Tensor):
                out_cat[field_id] = torch.cat([out_cat[field_id], out[i][field_id]], dim=0)
            elif isinstance(out[i][field_id], dict):
                for k, v in out[i][field_id].items():
                    if isinstance(v, np.ndarray):
                        out_cat[field_id][k] = np.concatenate([out_cat[field_id][k], out[i][field_id][k]], axis=0)
                    elif isinstance(v, torch.Tensor):
                        out_cat[field_id][k] = torch.cat([out_cat[field_id][k], out[i][field_id][k]], dim=0)
                    else:
                        out_cat[field_id][k].append(out[i][field_id][k])
            else:
                out_cat[field_id].append(out[i][field_id])

    if n_out_field == 1:
        return out_cat[0]
    else:
        return tuple(out_cat)


def get_batch_from_list(*args, start_idx, end_idx, gpu_on_func):
    """Get the batch of list from *args, each one can be an array or None

    Args:
        *args: containing arrays for input, assume each (B, d_in), torch or np array.
                Can be other params like float/str/bool/none, etc. But at least contains one array.
               If it is a dict, will slice each item and output the dict as well. (Do not use nested dict)
        start_idx: start idx for selection.
        end_idx: end idx for selection. Valid for value > B
        gpu_on_func: If True, will manually move torch tensor to 'gpu'. Original tensor should be on 'cpu'.

    Returns:
        out: list of torch or np array in (B, d_out)
        move_to_cpu: Whether need to move the tensor back to cpu. Only when they are on cpu before and moved to gpu
    """
    batch = []
    move_to_cpu = False
    for elem in args:
        if is_torch_or_np(elem):
            if isinstance(elem, torch.Tensor) and gpu_on_func and not elem.is_cuda:  # get to gpu
                move_to_cpu = True
                batch.append(elem[start_idx:end_idx].clone().cuda(non_blocking=True))  # move to 'gpu' manually
            else:
                batch.append(elem[start_idx:end_idx])
        elif isinstance(elem, dict):
            in_dict = {}
            for k, v in elem.items():
                if is_torch_or_np(v):
                    if isinstance(v, torch.Tensor) and gpu_on_func and not v.is_cuda:
                        move_to_cpu = True
                        in_dict[k] = v[start_idx:end_idx].clone().cuda(non_blocking=True)  # move to 'gpu' manually
                    else:
                        in_dict[k] = v[start_idx:end_idx]
                else:
                    in_dict[k] = v
            batch.append(in_dict)
        else:
            batch.append(elem)

    return batch, move_to_cpu


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
