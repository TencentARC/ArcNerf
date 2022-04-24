# -*- coding: utf-8 -*-

import os
import os.path as osp

import cv2
import numpy as np

from common.utils.img_utils import img_to_uint8
from common.utils.torch_utils import torch_to_np


def render_progress_imgs(inputs, output):
    """Actual render for progress image with label. It is perform in each step with a batch.
     Return a dict with list of image and filename. filenames should be irrelevant to progress
     Image should be in bgr with shape(h,w,3), which can be directly writen by cv.imwrite().
     Return None will not write anything.
     You should copy anything from input/output to forbid change of value
    """
    if inputs['H'][0] * inputs['W'][0] != inputs['img'].shape[1]:  # sampled rays, do not show anything
        return None

    dic = {}

    # for images
    names = []
    images = []
    idx = 0  # only sample from the first
    w, h = int(inputs['W'][idx]), int(inputs['H'][idx])
    # origin image, mask
    img = img_to_uint8(torch_to_np(inputs['img'][idx]).copy().reshape(h, w, 3))  # (H, W, 3)
    mask = torch_to_np(inputs['mask'][idx]).copy().reshape(h, w) if 'mask' in inputs else None  # (H, W)
    mask = (255.0 * mask).astype(np.uint8)[..., None].repeat(3, axis=-1) if mask is not None else None  # (H, W, 3)

    # pred rgb + img + error
    pred_rgb = ['rgb_coarse', 'rgb_fine', 'rgb']
    for pred_name in pred_rgb:
        if pred_name in output:
            pred_img = img_to_uint8(torch_to_np(output[pred_name][idx]).copy().reshape(h, w, 3))  # (H, W, 3)
            error_map = np.abs(img - pred_img)  # (H, W, 3)
            pred_cat = np.concatenate([img, pred_img, error_map], axis=1)  # (H, 3W, 3)
            names.append(pred_name)
            images.append(pred_cat)
    # depth, norm and put to uint8(0-255)
    pred_depth = ['depth_coarse', 'depth_fine', 'depth']
    for pred_name in pred_depth:
        if pred_name in output:
            pred_depth = torch_to_np(output[pred_name][idx]).copy().reshape(h, w)  # (H, W), 0~1
            pred_depth = (255.0 * pred_depth / (pred_depth.max() + 1e-8)).astype(np.uint8)
            pred_cat = np.concatenate([img, pred_depth[..., None].repeat(3, axis=-1)], axis=1)  # (H, 2W, 3)
            names.append(pred_name)
            images.append(pred_cat)
    # mask
    pred_mask = ['mask_coarse', 'mask_fine', 'mask']
    for pred_name in pred_mask:
        if pred_name in output:
            pred_mask = torch_to_np(output[pred_name][idx]).copy().reshape(h, w)  # (H, W), 0~1, obj area with 1
            pred_mask = (255.0 * pred_mask).astype(np.uint8)[..., None].repeat(3, axis=-1)  # (H, W, 3), 0~255
            mask_img = (255 - pred_mask) + img  # (H, W, 3), white bkg
            if mask is not None:
                error_map = (255.0 * np.abs(pred_mask - mask)).astype(np.uint8)  # (H, W, 3), 0~255
                pred_cat = np.concatenate([mask_img, pred_mask, error_map], axis=1)  # (H, 3W, 3)
            else:
                pred_cat = np.concatenate([mask_img, pred_mask], axis=1)  # (H, 2W, 3)
            names.append(pred_name)
            images.append(pred_cat)

    dic['imgs'] = {'names': names, 'imgs': images}

    return dic


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
        for img_name in files[0]['imgs']['names']:
            os.makedirs(osp.join(folder, img_name), exist_ok=True)

        for idx, file in enumerate(files):
            for name, img in zip(file['imgs']['names'], file['imgs']['imgs']):
                img_folder = osp.join(folder, name)
                os.makedirs(img_folder, exist_ok=True)

                if eval:
                    img_path = osp.join(img_folder, 'eval_{:04d}.png'.format(idx))
                else:
                    if num_sample == 1:
                        img_path = osp.join(
                            img_folder, 'epoch{:06d}_step{:05d}_global{:08d}.png'.format(epoch, step, global_step)
                        )
                    else:
                        img_path = osp.join(
                            img_folder,
                            'epoch{:06d}_step{:05d}_global{:08d}_{:04d}.png'.format(epoch, step, global_step, idx)
                        )

                cv2.imwrite(img_path, img)
