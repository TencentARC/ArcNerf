#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import os.path as osp

import numpy as np
import torch

from arcnerf.datasets import get_dataset
from arcnerf.geometry.poses import invert_poses
from arcnerf.geometry.projection import world_to_pixel
from arcnerf.geometry.volume import Volume
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.cfgs_utils import load_configs, valid_key_in_cfgs
from common.utils.img_utils import get_bbox_from_mask
from common.utils.logger import Logger
from common.utils.torch_utils import torch_to_np
from common.utils.video_utils import write_video
from common.visual.draw_cv2 import draw_bbox_on_img

# Some constant params
INIT_SIDE = 2.0
BBOX_EXPAND = 1.05
VOLUME_EXPAND = 1.25  # in case 2d bbox can not cover well
N_ITER = 1000
THRES = 1e-2
LR = 1e-2
WEIGHT_DECAY = 1e-3

# Result dir
RESULT_DIR = osp.abspath(osp.join(__file__, '../..', 'results', 'regress_volume'))
SAVE_VISUAL_RES = True
if SAVE_VISUAL_RES:
    os.makedirs(RESULT_DIR, exist_ok=True)


def generalized_iou_loss(gt_bboxes, pr_bboxes, reduction='mean'):
    """IOU Compare 2d bbox
    ref: https://github.com/CoinCheung/pytorch-loss/blob/master/generalized_iou_loss.py
    Args:
        gt_bboxes: tensor (-1, 4) min_x, min_y, max_x. max_y
        pr_bboxes: tensor (-1, 4) min_x, min_y, max_x. max_y
        reduction: reduction method
    """
    gt_area = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
    pr_area = (pr_bboxes[:, 2] - pr_bboxes[:, 0]) * (pr_bboxes[:, 3] - pr_bboxes[:, 1])

    # iou
    lt = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = gt_area + pr_area - inter
    iou = inter / union
    # enclosure
    lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    enclosure = wh[:, 0] * wh[:, 1]

    giou = iou - (enclosure - union) / enclosure
    loss = 1. - giou
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass

    return loss


if __name__ == '__main__':
    # parse args, logger
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, required=True, help='Configs yaml to be read')
    args, unknowns = parser.parse_known_args()
    cfgs = load_configs(args.configs, unknowns)
    logger = Logger()

    # get the train dataset
    dataset = get_dataset(cfgs.dataset, cfgs.dir.data_dir, mode='train', logger=logger)
    logger.add_log('Getting bbox for dataset {}'.format(cfgs.dataset.train.type))
    if valid_key_in_cfgs(cfgs.dataset.train, 'scale_radius'):
        logger.add_log('The scale radius is {}'.format(cfgs.dataset.train.scale_radius))
    dataset_type = cfgs.dataset.train.type
    scene_name = args.configs.split('/')[-1].split('.')[0]
    n_cam = len(dataset)

    # check whether silhouette/mask exist
    if 'mask' not in dataset[0]:
        logger.add_log('No silhouette provided for the dataset...exit', level='warning')
        exit()

    # init volume in world space
    volume = Volume(n_grid=None, side=INIT_SIDE, requires_grad=True)

    # get all camera pose and silhouette
    h, w = 0, 0
    masks = []
    intrinsics = []
    c2ws = []
    for data in dataset:
        h, w = data['H'], data['W']
        intrinsics.append(data['intrinsic'][None])  # (1, 3, 3)
        c2ws.append(data['c2w'][None])  # (1, 4, 4)
        masks.append(data['mask'].view(h, w).detach().numpy())  # (h, w)
    c2w = torch.cat(c2ws, dim=0)  # (N, 4, 4)
    w2c = invert_poses(c2w)  # (N, 4, 4)
    intrinsic = torch.cat(intrinsics, dim=0)  # (N, 3, 3)

    # get the 2d bbox from all mask
    bbox = []
    for mask in masks:
        # hard mask
        mask[mask > 0.5] = 1.0
        mask[mask < 0.5] = 0.0
        bbox.append(get_bbox_from_mask(mask, BBOX_EXPAND)[None])  # (1, 4)
    bbox = np.concatenate(bbox, axis=0)  # (N, 4)
    bbox_t = torch.from_numpy(bbox)

    # write the result file to video
    mask_with_bbox_file = osp.join(RESULT_DIR, 'mask_with_bbox_' + dataset_type + '_' + scene_name + '.mp4')
    mask_255 = [(m[..., None].repeat(3, -1) * 255.0).astype(np.uint8) for m in masks]
    mask_with_bbox = [draw_bbox_on_img(mask_255[i], bbox[i:i + 1]) for i in range(len(mask_255))]
    if SAVE_VISUAL_RES:
        write_video(mask_with_bbox, mask_with_bbox_file, fps=2)
        logger.add_log('Write mask with bbox into {}...'.format(mask_with_bbox_file))

    def get_bbox_from_volume(vol, cam_in, cam_w2c):
        """Get the 2d bbox from volume with cam"""
        vol.set_pts()  # in case volume changes
        corner = vol.get_corner()  # (8, 3)
        corner = torch.repeat_interleave(corner[None], cam_in.shape[0], 0)  # (N, 8, 3)
        corner_in_pixel = world_to_pixel(corner, cam_in, cam_w2c)  # (N, 8, 2)
        min_xy, _ = torch.min(corner_in_pixel, dim=1)  # (N, 2)
        max_xy, _ = torch.max(corner_in_pixel, dim=1)  # (N, 2)
        out = torch.cat([min_xy, max_xy], dim=-1)  # (N, 4)

        return out

    # run optimizer
    optimizer = torch.optim.Adam(volume.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for n_iter in range(N_ITER):
        reproj_bbox = get_bbox_from_volume(volume, intrinsic, w2c)
        iou_loss = generalized_iou_loss(reproj_bbox, bbox_t)
        optimizer.zero_grad()
        iou_loss.backward(retain_graph=True)
        optimizer.step()

        logger.add_log(
            'Iter {} - Loss {:.6f} - Bbox center {} - Bbox length {}'.format(
                n_iter, iou_loss,
                torch_to_np(volume.get_origin()).tolist(), volume.get_len()
            )
        )

        if iou_loss < THRES:
            break

    # expand the volume and get final bbox
    with torch.no_grad():
        volume.expand_len(VOLUME_EXPAND)
        pred_bbox = torch_to_np(get_bbox_from_volume(volume, intrinsic, w2c)).astype(bbox.dtype)  # (N, 4)
        mask_with_both_bbox_file = osp.join(
            RESULT_DIR, 'mask_with_both_bbox_' + dataset_type + '_' + scene_name + '.mp4'
        )
        mask_with_both_bbox = []
        for i in range(len(mask_255)):
            both_bbox = np.concatenate([bbox[i:i + 1], pred_bbox[i:i + 1]], axis=0)  # (2, 4)
            color = ['red', 'green']
            mask_with_both_bbox.append(draw_bbox_on_img(mask_255[i], both_bbox, color))
        if SAVE_VISUAL_RES:
            write_video(mask_with_both_bbox, mask_with_both_bbox_file, fps=2)
            logger.add_log('Write mask with bbox into {}...'.format(mask_with_both_bbox_file))

    logger.add_log('Final regress output :')
    logger.add_log('    Origin: {}'.format(torch_to_np(volume.get_origin())))
    logger.add_log('    xyz_len: {}'.format(volume.get_len()))

    # draw 3d output
    volume_3d_file = osp.join(RESULT_DIR, 'volume_' + dataset_type + '_' + scene_name + '.png')
    draw_3d_components(
        torch_to_np(c2w),
        intrinsic=torch_to_np(intrinsic[0]),
        volume={
            'grid_pts': torch_to_np(volume.get_corner()),
            'lines': volume.get_bound_lines(),
            'faces': volume.get_bound_faces()
        },
        sphere_radius=float(torch.max(torch.norm(c2w[:, :3, 3], dim=-1))),
        title='regress volume from training dataset',
        save_path=volume_3d_file,
        plotly=True,
        plotly_html=True
    )
