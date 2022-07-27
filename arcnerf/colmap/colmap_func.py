# -*- coding: utf-8 -*-

import os
import shutil
from subprocess import check_output

import numpy as np

from .colmap_lib import read_model
from .colmap_wrapper import run_colmap, run_colmap_dense
from common.utils.img_utils import get_n_img_in_dir, is_img_ext


def estimate_poses(scene_dir, logger, match_type, factors=None):
    """estimate poses by colmap. images are at scene_dir/images.

    Args:
        scene_dir: scene_dir contains image and poses
        logger: logger
        match_type: ['sequential_matcher', 'exhaustive_matcher']
        factors: list of int. Factor > 1 means smaller, <1 means larger

    Returns:
        Write .bins file and .npy files in scene_dir.
    """
    filenames = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(scene_dir, 'sparse/0')):
        cur_files = os.listdir(os.path.join(scene_dir, 'sparse/0'))
    else:
        cur_files = []

    if not all([f in cur_files for f in filenames]):
        logger.add_log('Need to run construct from colmap...')
        # sparse reconstruct from command line
        # TODO: may need to modify the database.db for known cam_poses
        run_colmap(scene_dir, logger, match_type)
    else:
        logger.add_log('No need to run reconstruction again...')

    # post processing
    logger.add_log('Post process sparse result...')
    poses, pts3d, perm, image_names = load_colmap_data(scene_dir, logger)

    # remove unregistered images
    image_ids_mapping = handle_unregistered_images(scene_dir, logger, poses, pts3d, image_names)

    save_poses(scene_dir, poses, pts3d, perm, logger, image_ids_mapping)

    if factors is not None:
        logger.add_log('Factor process by: {}'.format(factors))
        minify(scene_dir, logger, factors)

    logger.add_log('Finish processing in dir: {}'.format(scene_dir))


def dense_reconstruct(scene_dir, logger):
    """Dense reconstruct by colmap"""
    meshfile = os.path.join(scene_dir, 'dense/fused.ply')
    if not os.path.exists(meshfile):
        logger.add_log('Run dense reconstruction...')
        run_colmap_dense(scene_dir, logger)
    else:
        logger.add_log('Mesh file already at {}...'.format(meshfile))


def load_colmap_data(scene_dir, logger):
    """Load colmap sparse data from .bin files.

    Args:
       scene_dir: scene_dir contains image and poses
       logger: logger

    Returns:
        poses: w2c + focal, N camera
        pts3d: 3d point cloud in world coordinate
        perm: image order index by name
    """
    cam_data, img_data, pts3d = read_model(os.path.join(scene_dir, 'sparse/0'), '.bin')

    list_of_keys = list(cam_data.keys())
    cam = cam_data[list_of_keys[0]]
    logger.add_log('    Cameras: {} - Params {}'.format(cam.model, cam.params))

    h, w, f = cam.height, cam.width, cam.params[0]
    r_mats, t_mats = [], []

    names = [img_data[k].name for k in img_data]
    logger.add_log('    Images Num: {}'.format(len(names)))
    perm = np.argsort(names)
    for k in img_data:
        im = img_data[k]
        r_mats.append(im.qvec2rotmat())
        t_mats.append(im.tvec.reshape([3, 1]))

    r_mats = np.stack(r_mats, 0)  # (N, 3, 3)
    t_mats = np.stack(t_mats, 0)  # (N, 3, 1)

    poses_dict = {
        'R': r_mats,
        'T': t_mats,
        'h': h,
        'w': w,
        'f': f,
        'cam_type': cam.model,
        'cam_params': cam.params,
        'n_cam': r_mats.shape[0]
    }

    return poses_dict, pts3d, perm, names


def handle_unregistered_images(scene_dir, logger, poses, pts3d, image_names):
    """Colmap some times do not give pose estimation to some images.
     Move them to unreg_dir, and change the visible image_ids to remaining images.

    Args:
        scene_dir: scene_dir contains image and poses
        logger: logger
        poses: w2c + focal, N camera
        pts3d: 3d point cloud in world coordinate
        image_names: reg image list

    Returns:
        pts3d: with adjusted image_ids to remaining
    """
    unreg_dir = os.path.join(scene_dir, 'unreg_images')

    if os.path.exists(unreg_dir) and get_n_img_in_dir(unreg_dir):
        logger.add_log('Already move unreg images to {}'.format(unreg_dir))
        reg_image_names = [name for name in os.listdir(os.path.join(scene_dir, 'images')) if is_img_ext(name)]
        unreg_image_names = [name for name in os.listdir(unreg_dir) if is_img_ext(name)]
        all_image_names = reg_image_names + unreg_image_names

    elif poses['n_cam'] < get_n_img_in_dir(os.path.join(scene_dir, 'images')):
        os.makedirs(unreg_dir, exist_ok=True)
        logger.add_log(
            'Only get {}/{} images registered... Other images move to {}'.format(
                poses['n_cam'], get_n_img_in_dir(os.path.join(scene_dir, 'images')), unreg_dir
            )
        )
        all_image_names = [name for name in os.listdir(os.path.join(scene_dir, 'images')) if is_img_ext(name)]

    else:
        return None

    # new image mapping, assume the idx is given in python sorted order
    all_image_names = sorted(all_image_names)
    reg_image_name = sorted(image_names)
    image_ids_mapping = {}
    for i, name in enumerate(all_image_names):
        if name in reg_image_name:
            image_ids_mapping[i + 1] = reg_image_name.index(name) + 1

    # check old ids in pts3d
    old_image_ids = []
    for k in pts3d:
        old_image_ids.extend(pts3d[k].image_ids.tolist())
    old_image_ids = list(dict.fromkeys(old_image_ids))
    old_image_ids = sorted(old_image_ids)
    assert all([old_id in image_ids_mapping.keys() for old_id in old_image_ids]), 'Order not matched.., Please check'

    # move unreg files
    if poses['n_cam'] < get_n_img_in_dir(os.path.join(scene_dir, 'images')):
        unreg_image_names = [name for name in all_image_names if name not in image_names]
        unreg_image_files = [os.path.join(scene_dir, 'images', name) for name in unreg_image_names]
        dst_files = [os.path.join(unreg_dir, name) for name in unreg_image_names]
        for file, dst_file in zip(unreg_image_files, dst_files):
            shutil.move(file, dst_file)

    return image_ids_mapping


def save_poses(scene_dir, poses, pts3d, perm, logger, image_ids_mapping=None):
    """Save to npy file"""
    pts_arr = []
    vis_arr = []
    rgb_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        rgb_arr.append(pts3d[k].rgb)
        cams = [0] * poses['n_cam']  # N_cam
        for ind in pts3d[k].image_ids:
            if image_ids_mapping is not None:
                ind = image_ids_mapping[ind]
            if len(cams) < ind - 1:
                raise RuntimeError('ERROR: the correct camera poses for current points cannot be accessed')
            cams[ind - 1] = 1
        vis_arr.append(cams)
    pts_arr = np.array(pts_arr)
    rgb_arr = np.array(rgb_arr)
    vis_arr = np.array(vis_arr).transpose([1, 0])
    logger.add_log('    Points {}, Visibility {}'.format(pts_arr.shape, vis_arr.shape))

    # get z vals in cam coordinate
    w2c = np.concatenate([poses['R'], poses['T']], axis=-1)
    bottom = np.repeat(np.array([0, 0, 0, 1.]).reshape([1, 4])[None, ...], poses['n_cam'], axis=0)
    w2c = np.concatenate([w2c, bottom], axis=1)  # (N, 4, 4)
    pts_arr_homo = np.concatenate([pts_arr, np.ones(shape=(pts_arr.shape[0], 1))], axis=1).transpose([1, 0])
    pts_cam = np.matmul(w2c, pts_arr_homo)[:, :3, :]
    zvals = pts_cam[:, -1, :]
    valid_z = zvals[vis_arr == 1]
    logger.add_log('    Depth stats: min {}-max {}-mean {}'.format(valid_z.min(), valid_z.max(), valid_z.mean()))

    # in image/camera order, write w2c+f/w/h and zvals-range for each cam
    bounds = []
    for i in perm:
        vis = vis_arr[i, :]
        zs = zvals[i, :]
        zs = zs[vis == 1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        bounds.append([close_depth, inf_depth])
    bounds = np.stack(bounds, axis=0)

    # re-adjust to image_name order, and write all sparse result
    poses['R'] = poses['R'][perm, :, :]  # (N, 3, 3)
    poses['T'] = poses['T'][perm, :, :]  # (N, 3)
    poses['bounds'] = bounds  # (N, 2)
    poses['pts'] = pts_arr  # (Np, 3)
    poses['rgb'] = rgb_arr  # (Np, 3)
    poses['vis'] = vis_arr[perm]  # (N, Np)
    save_path = os.path.join(scene_dir, 'poses_bounds.npy')
    np.save(save_path, poses)
    logger.add_log('Write result to ' + save_path)


def minify(scene_dir, logger, factors=None, resolutions=None):
    """If the images are scaled by a factor in actual usage, need to change the cam poses
       This function only resize image, do not change the cam pose params.

       Args:
           scene_dir: scene_dir contains image and poses
           logger: logger
           factors: list of int. Factor > 1 means smaller, <1 means larger
           resolutions: list of tuple of resolution
    """
    if factors is None:
        factors = []
    if resolutions is None:
        factors = []

    need_to_load = False
    for r in factors:
        img_dir = os.path.join(scene_dir, 'images_{}'.format(r))
        if not os.path.exists(img_dir):
            need_to_load = True
    for r in resolutions:
        img_dir = os.path.join(scene_dir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(img_dir):
            need_to_load = True
    if not need_to_load:
        return

    img_dir = os.path.join(scene_dir, 'images')
    imgs = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    img_dir_ori = img_dir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resize_arg = '{}%'.format(int(100. / r))
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resize_arg = '{}x{}'.format(r[1], r[0])
        img_dir = os.path.join(scene_dir, name)
        if os.path.exists(img_dir):
            continue

        logger.add_log('    Minifying - factor {}, image {}'.format(r, img_dir))

        os.makedirs(img_dir, exist_ok=True)
        check_output('cp {}/* {}'.format(img_dir_ori, img_dir), shell=True)
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resize_arg, '-format', 'png', '*.{}'.format(ext)])
        os.chdir(img_dir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(img_dir, ext), shell=True)
