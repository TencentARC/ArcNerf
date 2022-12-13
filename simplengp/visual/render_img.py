# -*- coding: utf-8 -*-

import os
import os.path as osp

import cv2
import numpy as np

from common.utils.cfgs_utils import pop_none_item
from common.utils.img_utils import img_to_uint8
from common.utils.torch_utils import torch_to_np


def render_progress_imgs(inputs, output):
    """Actual render for progress image with label. It is perform in each step with a batch.
     Return a dict with list of image and filename. filenames should be irrelevant to progress
     Image should be in bgr with shape(h,w,3), which can be directly writen by cv.imwrite().
     Return None will not write anything.
     You should copy anything from input/output to forbid change of value
    """
    dic = {}
    if int(inputs['H'][0]) * int(inputs['W'][0]) == inputs['img'].shape[1]:  # val sample for full image
        dic['imgs'] = get_render_imgs(inputs, output)  # images

    # remove if all None, pop None item
    if all([v is None for v in dic.values()]):
        return None

    pop_none_item(dic)

    return dic


def get_render_imgs(inputs, output):
    """Get the render images with names"""
    names = []
    images = []
    idx = 0  # only sample from the first image

    w, h = int(inputs['W'][idx]), int(inputs['H'][idx])
    # origin image, mask
    img = img_to_uint8(torch_to_np(inputs['img'][idx]).copy().reshape(h, w, 3))  # (H, W, 3)
    mask = torch_to_np(inputs['mask'][idx]).copy().reshape(h, w) if 'mask' in inputs else None  # (H, W)
    mask = (255.0 * mask).astype(np.uint8)[..., None].repeat(3, axis=-1) if mask is not None else None  # (H, W, 3)

    # pred rgb + img + error
    pred_rgbs = ['rgb_coarse', 'rgb_fine', 'rgb']
    for pred_name in pred_rgbs:
        if pred_name in output:
            pred_img = img_to_uint8(torch_to_np(output[pred_name][idx]).copy().reshape(h, w, 3))  # (H, W, 3)
            error_map = cv2.applyColorMap(cv2.cvtColor(np.abs(img - pred_img), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
            pred_cat = np.concatenate([img, pred_img, error_map], axis=1)  # (H, 3W, 3)
            names.append(pred_name)
            images.append(pred_cat)

    # depth, norm and put to uint8(0-255)
    pred_depths = ['depth_coarse', 'depth_fine', 'depth']
    for pred_name in pred_depths:
        if pred_name in output:
            pred_depth = torch_to_np(output[pred_name][idx]).copy().reshape(h, w)  # (H, W), 0~1
            pred_depth = (255.0 * pred_depth / (pred_depth.max() + 1e-8)).astype(np.uint8)
            pred_cat = np.concatenate([img, pred_depth[..., None].repeat(3, axis=-1)], axis=1)  # (H, 2W, 3)
            names.append(pred_name)
            images.append(pred_cat)

    # mask
    pred_masks = ['mask_coarse', 'mask_fine', 'mask']
    for pred_name in pred_masks:
        if pred_name in output:
            pred_mask = torch_to_np(output[pred_name][idx]).copy().reshape(h, w)  # (H, W), 0~1, obj area with 1
            pred_mask = (255.0 * pred_mask).astype(np.uint8)[..., None].repeat(3, axis=-1)  # (H, W, 3), 0~255
            mask_img = 255 - pred_mask.copy().astype(np.uint16) + img.copy().astype(np.uint16)  # overflow
            mask_img = np.clip(mask_img, 0, 255).astype(np.uint8)  # (H, W, 3), white bkg
            if mask is not None:
                error_map = (255.0 * np.abs(pred_mask - mask)).astype(np.uint8)  # (H, W, 3), 0~255
                error_map = cv2.applyColorMap(cv2.cvtColor(error_map, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
                pred_cat = np.concatenate([mask_img, pred_mask, error_map], axis=1)  # (H, 3W, 3)
            else:
                pred_cat = np.concatenate([mask_img, pred_mask], axis=1)  # (H, 2W, 3)
            names.append(pred_name)
            images.append(pred_cat)

    img_dict = {'names': names, 'imgs': images}

    return img_dict


def write_progress_imgs(files, folder, epoch=None, step=None, global_step=None, eval=False):
    """Actual function to write the progress image from render image

    Args:
        files: a list of dict, each contains:
                imgs: with ['names', 'imgs'], each is the image and names
              You can also add other types of files (figs, etc) for rendering.
        folder: the main folder to save the result
        epoch: epoch, use when eval is False
        step: step, use when eval is False
        global_step: global_step, use when eval is True
        eval: If true, save name as 'eval_xxx.png', else 'epoch_step_global.png'
    """
    num_sample = len(files)

    # write the image
    if 'imgs' in files[0] and len(files[0]['imgs']['names']) > 0:
        for idx, file in enumerate(files):
            for name, img in zip(file['imgs']['names'], file['imgs']['imgs']):
                img_folder = osp.join(folder, name)
                os.makedirs(img_folder, exist_ok=True)
                img_path = get_dst_path(img_folder, eval, idx, num_sample, epoch, step, global_step)
                cv2.imwrite(img_path, img)


def get_dst_path(folder, eval, idx=None, num_sample=1, epoch=None, step=None, global_step=None):
    """Get the dist file name"""
    if eval:
        img_path = osp.join(folder, 'eval_{:04d}.png'.format(idx))
    else:
        if num_sample == 1:
            img_path = osp.join(folder, 'epoch{:06d}_step{:05d}_global{:08d}.png'.format(epoch, step, global_step))
        else:
            img_path = osp.join(
                folder, 'epoch{:06d}_step{:05d}_global{:08d}_{:04d}.png'.format(epoch, step, global_step, idx)
            )

    return img_path
