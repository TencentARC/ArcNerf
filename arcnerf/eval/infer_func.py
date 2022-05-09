# -*- coding: utf-8 -*-

import time
import os.path as osp

import numpy as np
import torch

from arcnerf.geometry.mesh import (
    extract_mesh,
    get_normals,
    get_face_centers,
    get_verts_by_faces,
    render_mesh_images,
    save_meshes,
    simplify_mesh,
)
from arcnerf.geometry.point_cloud import save_point_cloud
from arcnerf.geometry.poses import generate_cam_pose_on_sphere, invert_poses
from arcnerf.geometry.volume import Volume
from arcnerf.render.ray_helper import get_rays
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.cfgs_utils import valid_key_in_cfgs, get_value_from_cfgs_field
from common.utils.img_utils import img_to_uint8
from common.utils.torch_utils import torch_to_np
from common.utils.video_utils import write_video


def set_inference_data(cfgs, intrinsic, wh: tuple, dtype=torch.float32):
    """Set the inference data from cfgs(inference_part)

    Args:
        cfgs: cfgs of inference part containing render/volume
        intrinsic: (3, 3) np.array
        wh: image shape, generally from eval dataset, should be consistent with intrinsic
        dtype: dtype of rays input, by default torch.float32

    Returns:
        a dict with following parts. If any one is None, will not do render/extraction operation:
            render:
                cfgs: render cfgs
                wh: (W, H) of render image
                inputs: a list of inputs, each is list of n_cam images
                    each list contain n_cam dict with following keys:
                        c2w: novel view camera postion, (4, 4) np array
                        intrinsic: cam intrinsic, (3, 3) np array
                        rays_o: (1, HW, 3) torch tensor
                        rays_d: (1, HW, 3) torch tensor

            volume:
                cfgs: volume cfgs
                Vol: the Volume object from `geometry.volume`
    """
    assert intrinsic is not None, 'Please input an intrinsic of shape (3, 3)'
    assert len(wh) is not None, 'Please input correct image shape (w, h)'
    W, H = wh

    infer_data = {'render': None, 'volume': None}

    # parse inference cfgs
    render_cfgs = parse_render(cfgs)
    volume_cfgs = parse_volume(cfgs)

    # create data for rendering novel view
    if render_cfgs is not None:
        infer_data['render'] = {}
        infer_data['render']['cfgs'] = render_cfgs
        infer_data['render']['wh'] = (W, H)
        infer_data['render']['inputs'] = []

        for idx, mode in enumerate(render_cfgs['type']):
            c2w = generate_cam_pose_on_sphere(
                mode,
                render_cfgs['radius'],
                render_cfgs['n_cam'][idx],
                u_start=render_cfgs['u_start'],
                u_range=render_cfgs['u_range'],
                v_ratio=render_cfgs['v_ratio'],
                v_range=render_cfgs['v_range'],
                normal=render_cfgs['normal'],
                n_rot=render_cfgs['n_rot'],
                reverse=render_cfgs['reverse'],
                close=True
            )  # (n_cam, 4, 4), np array

            input = []
            for cam_id in range(c2w.shape[0]):
                ray_bundle = get_rays(
                    W,
                    H,
                    torch.tensor(intrinsic, dtype=dtype),
                    torch.tensor(c2w[cam_id], dtype=dtype),
                    wh_order=False,
                )  # (HW, 3) * 2
                input_per_img = {
                    'c2w': c2w[cam_id],  # (4, 4) np array
                    'intrinsic': intrinsic,  # (3, 3) np array
                    'rays_o': ray_bundle[0][None, :],  # (1, HW, 3) torch tensor
                    'rays_d': ray_bundle[1][None, :]  # (1, HW, 3) torch tensor
                }
                input.append(input_per_img)

            infer_data['render']['inputs'].append(input)

    # create volume for extraction
    if volume_cfgs is not None:
        infer_data['volume'] = {}
        infer_data['volume']['cfgs'] = volume_cfgs

        volume = Volume(
            volume_cfgs['n_grid'], volume_cfgs['origin'], volume_cfgs['side'], volume_cfgs['xlen'], volume_cfgs['ylen'],
            volume_cfgs['zlen']
        )
        infer_data['volume']['Vol'] = volume

        # add c2w and intrinsic for rendering geometry as well
        if volume_cfgs['render_mesh'] and render_cfgs is not None:
            infer_data['volume']['render'] = {
                'type': infer_data['render']['cfgs']['type'],
                'intrinsic': infer_data['render']['inputs'][0][0]['intrinsic'],  # (3, 3)
                'backend': volume_cfgs['render_backend'],
                'H': int(H),
                'W': int(W)
            }

            infer_data['volume']['render']['c2w'] = []
            for input in infer_data['render']['inputs']:
                c2w = []
                for input_per_img in input:
                    c2w.append(input_per_img['c2w'][None, ...])  # (4, 4)
                c2w = np.concatenate(c2w, axis=0)
                infer_data['volume']['render']['c2w'].append(c2w)  # (n_cam, 4, 4)

    # not doing anything, return None
    if infer_data['render'] is None and infer_data['volume'] is None:
        return None

    return infer_data


def parse_render(cfgs):
    """Return None if render is invalid in cfgs."""
    if not valid_key_in_cfgs(cfgs, 'render'):
        render_cfgs = None
    else:
        render_cfgs = {
            'type': get_value_from_cfgs_field(cfgs.render, 'type', ['circle', 'spiral']),
            'n_cam': get_value_from_cfgs_field(cfgs.render, 'n_cam', [30, 60]),
            'radius': get_value_from_cfgs_field(cfgs.render, 'radius', 3.0),
            'u_start': get_value_from_cfgs_field(cfgs.render, 'u_start', 0.0),
            'u_range': tuple(get_value_from_cfgs_field(cfgs.render, 'u_range', [0, 0.5])),
            'v_ratio': get_value_from_cfgs_field(cfgs.render, 'v_ratio', 0.0),
            'v_range': tuple(get_value_from_cfgs_field(cfgs.render, 'v_range', [-0.5, 0])),
            'n_rot': get_value_from_cfgs_field(cfgs.render, 'n_rot', 3),
            'normal': tuple(get_value_from_cfgs_field(cfgs.render, 'normal', [0.0, 1.0, 0.0])),
            'reverse': get_value_from_cfgs_field(cfgs.render, 'reverse', False),
            'fps': get_value_from_cfgs_field(cfgs.render, 'fps', 5)
        }

        assert len(render_cfgs['type']) == len(render_cfgs['n_cam']), 'Inconsistent mode and n_cam num'
        assert len(render_cfgs['u_range']) == 2, 'Please input u_range as list of 2'
        assert len(render_cfgs['v_range']) == 2, 'Please input v_range as list of 2'
        assert len(render_cfgs['normal']) == 3, 'Please input normal as list of 3'

    return render_cfgs


def parse_volume(cfgs):
    """Return None if volume is invalid in cfgs"""
    if not valid_key_in_cfgs(cfgs, 'volume'):
        volume_cfgs = None
    else:
        volume_cfgs = {
            'n_grid': get_value_from_cfgs_field(cfgs.volume, 'n_grid', 128),
            'origin': tuple(get_value_from_cfgs_field(cfgs.volume, 'origin', [0.0, 0.0, 0.0])),
            'xlen': get_value_from_cfgs_field(cfgs.volume, 'xlen', None),
            'ylen': get_value_from_cfgs_field(cfgs.volume, 'ylen', None),
            'zlen': get_value_from_cfgs_field(cfgs.volume, 'zlen', None),
            'level': get_value_from_cfgs_field(cfgs.volume, 'level', 50.0),
            'grad_dir': get_value_from_cfgs_field(cfgs.volume, 'grad_dir', 'descent'),
            'render_mesh': valid_key_in_cfgs(cfgs.volume, 'render_mesh'),
            'render_backend': get_value_from_cfgs_field(cfgs.volume.render_mesh, 'backend'),
        }
        if any([length is None for length in [volume_cfgs['xlen'], volume_cfgs['ylen'], volume_cfgs['zlen']]]):
            volume_cfgs['side'] = get_value_from_cfgs_field(volume_cfgs, 'side', 1.5)  # make sure volume exist
        else:
            volume_cfgs['side'] = get_value_from_cfgs_field(volume_cfgs, 'side', None)

    return volume_cfgs


@torch.no_grad()
def run_infer(data, get_model_feed_in, model, logger, device):
    """Run inference for novel view rendering and mesh extraction
       We turn it to eval() mode at beginning but don't turn it to train() mode at the end,
       you need to do it outside this func if needed

       The mesh extraction are all in cpu now. In the future, may need gpu method.

    Returns:
        files: dict with following keys:
            render: list of item.
                    each item is a list of render img in (H, W, 3) in bgr order
            volume: extracted verts and colors.
                corner: get (8,3) corner pts
                bound_lines: get 12 bound_lines for volume
                pc: with valid pts and color
                mesh: with verts, face, normal and color
    """
    model.eval()
    files = {'render': None, 'volume': None}

    # render image
    if data['render'] is not None and len(data['render']['inputs']) > 0:
        files['render'] = run_infer_render(data['render'], get_model_feed_in, model, device, logger)

    # extract volume/mesh
    if data['volume'] is not None:
        files['volume'] = run_infer_volume(data['volume'], model, device, logger)

    # not doing anything, return None
    if files['render'] is None and files['volume'] is None:
        return None

    return files


@torch.no_grad()
def run_infer_render(data, get_model_feed_in, model, device, logger):
    """Run render inference and return lists of different video frames"""
    render_out = []
    for idx, input in enumerate(data['inputs']):
        total_forward_time = 0.0
        logger.add_log('Rendering video {}...'.format(idx))
        img_w, img_h = int(data['wh'][0]), int(data['wh'][1])
        images = []
        for rays in input:
            feed_in, batch_size = get_model_feed_in(rays, 'cpu')  # only read rays_o/d here, (1, WH, 3)
            assert batch_size == 1, 'Only one image is sent to model at once for inference...'

            time0 = time.time()
            output = model(feed_in, inference_only=True)
            total_forward_time += (time.time() - time0)

            # get rgb only
            rgb_key = [key for key in output if key.startswith('rgb')]
            assert len(rgb_key) == 1, 'Only one rgb value should be produced by model in inference mode...'
            rgb = output[rgb_key[0]]  # (1, HW, 3)
            rgb = img_to_uint8(torch_to_np(rgb).copy()).reshape(img_h, img_w, 3)  # (H, W, 3), bgr
            images.append(rgb)
        render_out.append(images)

        logger.add_log(
            '    Render {} image, each hw({}/{}) total time {:.2f}s'.format(
                len(input), img_h, img_w, total_forward_time
            )
        )
        logger.add_log('    Each image takes time {:.2f}s'.format(total_forward_time / float(len(input))))

    return render_out


@torch.no_grad()
def run_infer_volume(data, model, device, logger, max_pts=200000, max_faces=500000):
    """Run volume inference and return pts and mesh
    Reduce pts and faces num for visual in plotly/plt
    # TODO: extract from sigma now. we may need to extract from model if model support
    """
    volume_out = {}
    logger.add_log('Extracting volume from model...')
    volume = data['Vol']
    n_grid = data['cfgs']['n_grid']
    level = data['cfgs']['level']
    grad_dir = data['cfgs']['grad_dir']
    volume_pts = volume.get_volume_pts()  # (n_grid^3, 3) pts in torch
    volume_size = volume.get_volume_size()  # (3,) tuple
    volume_len = volume.get_len()  # (3,) tuple

    # for volume visual
    volume_out['corner'] = torch_to_np(volume.get_corner())
    volume_out['bound_lines'] = volume.get_bound_lines()

    # use zeros dir to represent volume pts dir
    time0 = time.time()
    sigma, rgb = model.forward_pts_dir(volume_pts, None)
    sigma, rgb = torch_to_np(sigma), torch_to_np(rgb)
    logger.add_log('    Forward {}^3 time for model is {:.2f}s'.format(n_grid, time.time() - time0))
    logger.add_log('    Sigma value range {:.2f}-{:.2f}'.format(sigma.min(), sigma.max()))

    # for 3d point cloud visual, valid sigma is area with large enough sigma
    valid_sigma = (sigma >= level)  # (n^3,)
    volume_out['pc'] = None
    volume_out['mesh'] = None

    if not np.any(valid_sigma):
        logger.add_log('You do not have any valid extracting, please check the isolevel', level='warning')
    else:
        volume_out['pc'] = {}
        valid_pts = torch_to_np(volume_pts)[valid_sigma]  # (n_valid, 3)
        valid_rgb = rgb[valid_sigma]  # (n_valid, 3)
        n_pts = valid_pts.shape[0]
        logger.add_log('    Getting {} valid pts'.format(n_pts))
        volume_out['pc']['full'] = {'pts': valid_pts.copy(), 'color': valid_rgb.copy()}
        # sample for plot
        if n_pts > max_pts:
            logger.add_log('    Sample to {} pts'.format(max_pts))
            choice = np.random.choice(range(n_pts), max_pts, replace=False)
            valid_pts = valid_pts[choice]
            valid_rgb = valid_rgb[choice]
        volume_out['pc']['sample'] = {'pts': valid_pts.copy(), 'color': valid_rgb.copy()}

    if np.any(valid_sigma):
        sigma = sigma.reshape((n_grid, n_grid, n_grid))  # (n, n, n)

        try:
            # full mesh
            time0 = time.time()
            verts, faces, _ = extract_mesh(sigma.copy(), level, volume_size, volume_len, grad_dir)
            logger.add_log('    Extract mesh time {:.2f}s'.format(time.time() - time0))
            logger.add_log('    Extract {} verts, {} faces'.format(verts.shape[0], faces.shape[0]))

            volume_out['mesh'] = {}
            volume_out['mesh']['full'] = get_mesh_components(verts, faces, model, volume_pts.dtype, logger)

            # simplify mesh if need
            time0 = time.time()
            if faces.shape[0] <= max_faces:  # do not need to run again
                volume_out['mesh']['simplify'] = volume_out['mesh']['full']
            else:
                verts_sim, faces_sim = simplify_mesh(verts, faces, max_faces)
                logger.add_log('    Simplify mesh time {:.2f}s'.format(time.time() - time0))
                logger.add_log('    Simplify {} verts, {} faces'.format(verts_sim.shape[0], faces_sim.shape[0]))
                volume_out['mesh']['simplify'] = get_mesh_components(
                    verts_sim, faces_sim, model, volume_pts.dtype, logger
                )

            # add c2w and intrinsic
            if 'render' in data:
                volume_out['mesh']['render'] = {
                    'type': data['render']['type'],
                    'H': data['render']['H'],
                    'W': data['render']['W'],
                    'c2w': data['render']['c2w'],
                    'intrinsic': data['render']['intrinsic'],
                    'backend': data['render']['backend'],
                    'device': next(model.parameters()).device  # to allow gpu usage for rendering
                }

        except ValueError:
            logger.add_log('Can not extract mesh from volue', level='warning')

    if volume_out['pc'] is None and volume_out['mesh'] is None:
        return None

    return volume_out


def get_mesh_components(verts, faces, model, dtype, logger):
    """Get all the mesh components using model"""
    vert_normals, face_normals = get_normals(verts, faces)
    face_centers = get_face_centers(verts, faces)
    n_verts, n_faces = verts.shape[0], faces.shape[0]

    # get vert_colors, view point is the reverse normal
    vert_view_dir = -vert_normals
    vert_pts = torch.tensor(verts, dtype=dtype)  # (n, 3)
    vert_view_dir = torch.tensor(vert_view_dir, dtype=dtype)  # (n, 3)

    time0 = time.time()
    _, vert_colors = model.forward_pts_dir(vert_pts, vert_view_dir)
    vert_colors = torch_to_np(vert_colors)
    logger.add_log('    Get verts color for all {} verts takes {:.2f}s'.format(n_verts, time.time() - time0))

    # get face_colors, view point is the reverse normal
    face_view_dir = -face_normals
    face_center_pts = torch.tensor(face_centers, dtype=dtype)  # (n, 3)
    face_view_dir = torch.tensor(face_view_dir, dtype=dtype)  # (n, 3)

    time0 = time.time()
    _, face_colors = model.forward_pts_dir(face_center_pts, face_view_dir)
    face_colors = torch_to_np(face_colors)
    logger.add_log('    Get faces color for all {} faces takes {:.2f}s'.format(n_faces, time.time() - time0))

    res = {
        'verts': verts,
        'faces': faces,
        'vert_normals': vert_normals,
        'face_normals': face_normals,
        'face_centers': face_centers,
        'vert_colors': vert_colors,
        'face_colors': face_colors,
    }

    return res


def write_infer_files(files, folder, data, logger):
    """Write infer result to the folder. Only show limit number of pts and mesh"""
    if files is None:
        logger.add_log('No inference perform...', level='warning')
        return

    # write down video
    if files['render'] is not None:
        for vid, frames in enumerate(files['render']):
            video_path = osp.join(
                folder, 'render_video{}_{}_n{}_fps{}.mp4'.format(
                    vid, data['render']['cfgs']['type'][vid], data['render']['cfgs']['n_cam'][vid],
                    data['render']['cfgs']['fps']
                )
            )
            write_video(frames, video_path, fps=data['render']['cfgs']['fps'])
        logger.add_log('Write videos to {}'.format(folder))

    # write down extract mesh
    if files['volume'] is not None:
        corner = files['volume']['corner']
        bound_lines = files['volume']['bound_lines']
        volume_dict = {'grid_pts': corner, 'lines': bound_lines}
        if 'pc' in files['volume']:
            pc = files['volume']['pc']
            # full pts to ply
            pts = pc['full']['pts']
            pts_colors = pc['full']['color']
            pc_file = osp.join(folder, 'pc_extract.ply')
            save_point_cloud(pc_file, pts, pts_colors)
            # sample pts to plotly
            pts = pc['sample']['pts']
            pts_colors = pc['sample']['color']
            # draw pts in plotly
            file_path = osp.join(folder, 'pc_extract.png')
            draw_3d_components(
                points=pts,
                point_colors=pts_colors,
                point_size=10,
                volume=volume_dict,
                title='valid pts({}) from volume'.format(pts.shape[0]),
                save_path=file_path,
                plotly=True,
                plotly_html=True
            )
            logger.add_log('Write point cloud visual to {}'.format(folder))

        if 'mesh' in files['volume']:
            mesh = files['volume']['mesh']

            # save full mesh as .ply file, save verts/faces only
            mesh_file = osp.join(folder, 'mesh_extract.ply')
            save_meshes(
                mesh_file, mesh['full']['verts'], mesh['full']['faces'], mesh['full']['vert_colors'],
                mesh['full']['face_colors'], mesh['full']['vert_normals'], mesh['full']['face_normals']
            )
            mesh_geo_file = osp.join(folder, 'mesh_extract_geo.ply')
            save_meshes(mesh_geo_file, mesh['full']['verts'], mesh['full']['faces'], geo_only=True)

            # draw simplified in plotly
            verts_sim, faces_sim = mesh['simplify']['verts'], mesh['simplify']['faces']
            face_colors_sim = mesh['simplify']['face_colors']
            verts_by_faces, _ = get_verts_by_faces(verts_sim, faces_sim, None)
            file_path = osp.join(folder, 'mesh_extract.png')
            draw_3d_components(
                volume=volume_dict,
                meshes=[verts_by_faces],
                face_colors=[face_colors_sim],
                title='Meshes ({} faces) extract from volume'.format(verts_by_faces.shape[0]),
                save_path=file_path,
                plotly=True,
                plotly_html=True
            )
            logger.add_log('Write mesh visual to {}'.format(folder))

            if 'render' in mesh:  # render the mesh only
                render = mesh['render']
                for type, c2w in zip(render['type'], render['c2w']):
                    color_imgs = render_mesh_images(
                        mesh['full']['verts'],
                        mesh['full']['faces'],
                        mesh['full']['vert_colors'],
                        mesh['full']['face_colors'],
                        mesh['full']['vert_normals'],
                        mesh['full']['face_normals'],
                        render['H'],
                        render['W'],
                        invert_poses(c2w),
                        render['intrinsic'],
                        render['backend'],
                        device=render['device']
                    )  # (n_cam, h, w, 3)

                    file_path = osp.join(folder, 'color_mesh_render_type{}.mp4'.format(type))
                    write_video([color_imgs[idx] for idx in range(color_imgs.shape[0])], file_path, True)

                    geo_imgs = render_mesh_images(
                        mesh['full']['verts'],
                        mesh['full']['faces'],
                        None,
                        None,
                        mesh['full']['vert_normals'],
                        mesh['full']['face_normals'],
                        render['H'],
                        render['W'],
                        invert_poses(c2w),
                        render['intrinsic'],
                        render['backend'],
                        device=render['device']
                    )  # (n_cam, h, w, 3)

                    file_path = osp.join(folder, 'geo_mesh_render_type{}.mp4'.format(type))
                    write_video([geo_imgs[idx] for idx in range(geo_imgs.shape[0])], file_path, True)
