# -*- coding: utf-8 -*-

import os
import os.path as osp

import cv2
import numpy as np

from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.render.ray_helper import sample_ray_marching_output_by_index
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.img_utils import img_to_uint8
from common.utils.torch_utils import torch_to_np, np_wrapper
from common.visual import get_colors
from common.visual.plot_2d import draw_2d_components


def render_progress_imgs(inputs, output):
    """Actual render for progress image with label. It is perform in each step with a batch.
     Return a dict with list of image and filename. filenames should be irrelevant to progress
     Image should be in bgr with shape(h,w,3), which can be directly writen by cv.imwrite().
     Return None will not write anything.
     You should copy anything from input/output to forbid change of value
    """
    if int(inputs['H'][0]) * int(inputs['W'][0]) == inputs['img'].shape[1]:  # val sample for full image
        dic = {
            'imgs': get_render_imgs(inputs, output),  # images
            'rays': get_sample_ray_imgs(inputs, output)  # ray
        }
    else:  # sample some rays for visual
        dic = {'rays': get_sample_ray_imgs(inputs, output, train=True)}

    # remove none dict
    if all([v is None for v in dic.values()]):
        return None

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

    img_dict = {'names': names, 'imgs': images}

    return img_dict


def get_sample_ray_imgs(inputs, output, train=False, sample_num=16):
    """Get the sample rays image"""
    if not any([k.startswith('progress_') for k in output.keys()]):
        return None

    idx = 0  # only sample from the first image
    w, h = int(inputs['W'][idx]), int(inputs['H'][idx])
    n_rays = inputs['rays_o'].shape[1]

    if train:  # train mode, sample some rays
        ray_index = np.random.choice(range(n_rays), sample_num, replace=False).tolist()
        ray_id = ['sample_{}'.format(idx) for idx in range(n_rays)]
    else:  # valid mode, the top left and center rays, represent background and object
        ray_index = [0, int(w / 2.0 * h + h / 2.0), n_rays - 1]  # list of index   TODO: Check this at center
        ray_id = ['lefttop', 'center', 'rightdown']

    sample_dict = {
        'index': ray_index,
        'id': ray_id,
        'rays_o': [],
        'rays_d': [],
    }
    # get the progress keys
    prgress_keys = []
    for key in output.keys():
        if key.startswith('progress_'):
            prgress_keys.append(key)
    # origin, rays with progress
    rays_o = torch_to_np(inputs['rays_o'][idx]).copy()  # (n_rays, 3)
    rays_d = torch_to_np(inputs['rays_d'][idx]).copy()  # (n_rays, 3)
    progress = {}
    for key in prgress_keys:
        progress[key] = torch_to_np(output[key][idx]).copy()  # (n_rays, n_pts)
        sample_dict[key.replace('progress_', '')] = []

    for index in ray_index:
        sample_dict['rays_o'].append(rays_o[index, :][None, :])  # (1, 3)
        sample_dict['rays_d'].append(rays_d[index, :][None, :])  # (1, 3)
        for key in prgress_keys:
            sample_dict[key.replace('progress_', '')].append(progress[key][index, :][None, :])  # (1, n_pts)

    for key in sample_dict.keys():
        if key != 'index' and key != 'id':
            sample_dict[key] = np.concatenate(sample_dict[key], axis=0)  # (n_idx, 3/n_pts)

    # normal sigma to (0, 1) for visual 3d
    pts_size = None
    if 'sigma' in sample_dict.keys():
        sigma = sample_dict['sigma'].copy()  # (n_idx, n_pts)
        pts_size = (sigma - sigma.min(1)[:, None]) / (sigma.max(1)[:, None] - sigma.min(1)[:, None])
        pts_size *= 50.0

    # output
    ray_dict = {}
    # 3d ray status, draw all the rays with point together
    pts = np_wrapper(get_ray_points_by_zvals, sample_dict['rays_o'], sample_dict['rays_d'],
                     sample_dict['zvals']).reshape(-1, 3)  # (n_id, n_pts, 3)
    ray_dict['3d'] = {
        'points': pts,  # (n_id * n_pts)
        'point_size': pts_size.reshape(-1) if pts_size is not None else None,  # (n_id * n_pts)
        'point_colors': get_colors('red', to_np=True),
        'rays': (sample_dict['rays_o'], sample_dict['rays_d'] * sample_dict['zvals'].max(1)[:, None])
    }
    # 2d ray change for val mode only
    if not train:
        ray_dict['2d'] = {}
        ray_dict['2d']['samples'] = sample_ray_marching_output_by_index(
            sample_dict,
            index=range(len(sample_dict['index'])),
        )[0]
        ray_dict['2d']['names'] = sample_dict['id']

    return ray_dict


def write_progress_imgs(
    files, folder, epoch=None, step=None, global_step=None, eval=False, radius=None, volume_dict=None
):
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
        radius: if not None, draw the bounding radius for 3d rays
        volume_dict: if not None, draw the volume for 3d rays
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

    # write the rays by plotly
    if 'rays' in files[0] and files[0]['rays'] is not None:
        if '3d' in files[0]['rays']:
            rays_folder = osp.join(folder, 'rays_3d')
            os.makedirs(rays_folder, exist_ok=True)

            for idx, file in enumerate(files):
                rays_3d = file['rays']['3d']
                img_path = get_dst_path(rays_folder, eval, idx, num_sample, epoch, step, global_step)
                draw_3d_components(
                    **rays_3d,
                    sphere_radius=radius,
                    volume=volume_dict,
                    title='3d rays. pts size proportional to sigma',
                    save_path=img_path,
                    plotly=True,
                    plotly_html=True
                )

        if '2d' in files[0]['rays']:
            rays_folder = osp.join(folder, 'rays_2d')
            os.makedirs(rays_folder, exist_ok=True)
            for idx, file in enumerate(files):
                for name, rays_2d in zip(file['rays']['2d']['names'], file['rays']['2d']['samples']):
                    ray_folder = osp.join(rays_folder, '{}'.format(name))
                    os.makedirs(ray_folder, exist_ok=True)
                    img_path = get_dst_path(ray_folder, eval, idx, num_sample, epoch, step, global_step)
                    draw_2d_components(
                        **rays_2d,
                        title='2d rays, id: {}'.format(name),
                        save_path=img_path,
                    )


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
