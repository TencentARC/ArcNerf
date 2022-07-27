# -*- coding: utf-8 -*-

import time
import os
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


class Inferencer(object):
    """An inferencer to infer on the model"""

    def __init__(self, cfgs, intrinsic, wh: tuple, device, logger, dtype=torch.float32):
        """Set the inference data from cfgs(inference_part)

            Args:
                cfgs: cfgs of inference part containing render/volume
                intrinsic: (3, 3) np.array
                wh: image shape, generally from eval dataset, should be consistent with intrinsic
                device: device used
                logger: logger used
                dtype: dtype of rays input, by default torch.float32
            """
        assert intrinsic is not None and intrinsic.shape == (3, 3), 'Please input an intrinsic of shape (3, 3)'
        assert len(wh) is not None, 'Please input correct image shape (w, h)'
        self.W, self.H = wh
        self.intrinsic = intrinsic
        self.dtype = dtype
        self.device = device
        self.logger = logger

        self.cfgs = cfgs
        # parse render/volume cfgs
        self.render_cfgs = self.parse_render_cfgs()
        self.volume_cfgs = self.parse_volume_cfgs()

        # get render/volume preparation data like camera_path, volume, etc
        self.render_data = self.set_render_data()
        self.volume_data = self.set_volume_data()

    def is_none(self):
        """Return None if render_data and volume_data are none"""
        if self.render_data is None and self.volume_data is None:
            return True
        else:
            return False

    def get_wh(self):
        """Get the image width and height"""
        return self.W, self.H

    def get_intrinsic(self):
        """Get the intrinsic"""
        return self.intrinsic

    def parse_render_cfgs(self):
        """ Get the configs for rendering.
        If render in (cfgs.inference).render, return a dict with default settings.
        Return None if render is invalid in cfgs.
        """
        if not valid_key_in_cfgs(self.cfgs, 'render'):
            render_cfgs = None
        else:
            render_cfgs = {
                'type': get_value_from_cfgs_field(self.cfgs.render, 'type', ['circle', 'spiral']),
                'n_cam': get_value_from_cfgs_field(self.cfgs.render, 'n_cam', [30, 60]),
                'radius': get_value_from_cfgs_field(self.cfgs.render, 'radius', 3.0),
                'u_start': get_value_from_cfgs_field(self.cfgs.render, 'u_start', 0.0),
                'u_range': tuple(get_value_from_cfgs_field(self.cfgs.render, 'u_range', [0, 0.5])),
                'v_ratio': get_value_from_cfgs_field(self.cfgs.render, 'v_ratio', 0.0),
                'v_range': tuple(get_value_from_cfgs_field(self.cfgs.render, 'v_range', [-0.5, 0])),
                'n_rot': get_value_from_cfgs_field(self.cfgs.render, 'n_rot', 3),
                'normal': tuple(get_value_from_cfgs_field(self.cfgs.render, 'normal', [0.0, 1.0, 0.0])),
                'reverse': get_value_from_cfgs_field(self.cfgs.render, 'reverse', False),
                'fps': get_value_from_cfgs_field(self.cfgs.render, 'fps', 5),
                'surface_render': get_value_from_cfgs_field(self.cfgs.render, 'surface_render', None)
            }
            render_cfgs['repeat'] = get_value_from_cfgs_field(
                self.cfgs.render, 'repeat', [1] * len(render_cfgs['n_cam'])
            )

            assert len(render_cfgs['type']) == len(render_cfgs['n_cam']), 'Inconsistent mode and n_cam num'
            assert len(render_cfgs['u_range']) == 2, 'Please input u_range as list of 2'
            assert len(render_cfgs['v_range']) == 2, 'Please input v_range as list of 2'
            assert len(render_cfgs['normal']) == 3, 'Please input normal as list of 3'
            assert len(render_cfgs['repeat']) == len(render_cfgs['n_cam']), 'Please make correct repeat num'

        return render_cfgs

    def parse_volume_cfgs(self):
        """ Get the configs for volume.
        If render in (cfgs.inference).volume, return a dict with default settings.
        Return None if volume is invalid in cfgs.
        """
        if not valid_key_in_cfgs(self.cfgs, 'volume'):
            volume_cfgs = None
        else:
            volume_cfgs = {
                'n_grid': get_value_from_cfgs_field(self.cfgs.volume, 'n_grid', 128),
                'origin': tuple(get_value_from_cfgs_field(self.cfgs.volume, 'origin', [0.0, 0.0, 0.0])),
                'xlen': get_value_from_cfgs_field(self.cfgs.volume, 'xlen', None),
                'ylen': get_value_from_cfgs_field(self.cfgs.volume, 'ylen', None),
                'zlen': get_value_from_cfgs_field(self.cfgs.volume, 'zlen', None),
                'level': get_value_from_cfgs_field(self.cfgs.volume, 'level', 50.0),
                'grad_dir': get_value_from_cfgs_field(self.cfgs.volume, 'grad_dir', 'descent'),
                'chunk_pts_factor': get_value_from_cfgs_field(self.cfgs.volume, 'chunk_pts_factor', 1),
                'render_mesh': valid_key_in_cfgs(self.cfgs.volume, 'render_mesh'),
                'render_backend': get_value_from_cfgs_field(self.cfgs.volume.render_mesh, 'backend'),
            }
            if any([length is None for length in [volume_cfgs['xlen'], volume_cfgs['ylen'], volume_cfgs['zlen']]]):
                volume_cfgs['side'] = get_value_from_cfgs_field(self.cfgs.volume, 'side', 1.5)  # make sure volume exist
            else:
                volume_cfgs['side'] = get_value_from_cfgs_field(self.cfgs.volume, 'side', None)

        return volume_cfgs

    def get_render_cfgs(self, key=None):
        """Get the render cfgs"""
        if key is None:
            return self.render_cfgs
        return self.render_cfgs[key]

    def get_volume_cfgs(self, key=None):
        """Get the volume cfgs"""
        if key is None:
            return self.volume_cfgs
        return self.volume_cfgs[key]

    def set_render_data(self):
        """Set data for renderer. Including cam path and surface_render. Return a dict contain necessary info.
        Returns:
            render:
                inputs: a list of inputs, each is list of n_cam images
                    each list contain n_cam dict with following keys:
                        c2w: novel view camera position, (4, 4) np array
                        intrinsic: cam intrinsic, (3, 3) np array
                        rays_o: (1, HW, 3) torch tensor
                        rays_d: (1, HW, 3) torch tensor
                        rays_r: (1, HW, 1) torch tensor
            Return None if not set configs.
        """
        render_data = None
        if self.render_cfgs is not None:
            render_data = {'inputs': []}

            for idx, mode in enumerate(self.render_cfgs['type']):
                c2w = generate_cam_pose_on_sphere(
                    mode,
                    self.render_cfgs['radius'],
                    self.render_cfgs['n_cam'][idx],
                    u_start=self.render_cfgs['u_start'],
                    u_range=self.render_cfgs['u_range'],
                    v_ratio=self.render_cfgs['v_ratio'],
                    v_range=self.render_cfgs['v_range'],
                    normal=self.render_cfgs['normal'],
                    n_rot=self.render_cfgs['n_rot'],
                    reverse=self.render_cfgs['reverse'],
                    close=True
                )  # (n_cam, 4, 4), np array

                input = []
                for cam_id in range(c2w.shape[0]):
                    ray_bundle = get_rays(
                        self.W,
                        self.H,
                        torch.tensor(self.intrinsic, dtype=self.dtype),
                        torch.tensor(c2w[cam_id], dtype=self.dtype),
                        wh_order=False,
                    )  # (HW, 3) * 2
                    input_per_img = {
                        'c2w': c2w[cam_id],  # (4, 4) np array
                        'intrinsic': self.intrinsic,  # (3, 3) np array
                        'rays_o': ray_bundle[0][None, :],  # (1, HW, 3) torch tensor
                        'rays_d': ray_bundle[1][None, :],  # (1, HW, 3) torch tensor
                        'rays_r': ray_bundle[3][None, :]  # (1, HW, 1) torch tensor
                    }
                    input.append(input_per_img)

                render_data['inputs'].append(input)

            # add surface render result
            if self.render_cfgs['surface_render'] is not None:
                render_data['surface_render'] = self.render_cfgs['surface_render'].__dict__

        return render_data

    def get_render_data(self):
        """Get the render data"""
        return self.render_data

    def set_volume_data(self):
        """Set data for volume extraction. Including volume and camera_path for rendering
        Returns:
            volume:
                cfgs: volume cfgs
                Vol: the Volume object from `geometry.volume`
            Return None if not set configs.
        """
        volume_data = None
        if self.volume_cfgs is not None:
            volume_data = {}

            volume = Volume(
                self.volume_cfgs['n_grid'],
                self.volume_cfgs['origin'],
                self.volume_cfgs['side'],
                self.volume_cfgs['xlen'],
                self.volume_cfgs['ylen'],
                self.volume_cfgs['zlen'],
                dtype=self.dtype
            )
            volume_data['Vol'] = volume

            # add c2w and intrinsic for rendering geometry as well
            if self.volume_cfgs['render_mesh'] and self.render_cfgs is not None:
                volume_data['render'] = {
                    'type': self.render_cfgs['type'],
                    'intrinsic': self.intrinsic,  # (3, 3)
                    'backend': self.volume_cfgs['render_backend'],
                    'H': int(self.H),
                    'W': int(self.W),
                    'fps': self.render_cfgs['fps'],
                    'repeat': self.render_cfgs['repeat']
                }

                volume_data['render']['c2w'] = []
                for input in self.render_data['inputs']:
                    c2w = []
                    for input_per_img in input:
                        c2w.append(input_per_img['c2w'][None, ...])  # (4, 4)
                    c2w = np.concatenate(c2w, axis=0)
                    volume_data['render']['c2w'].append(c2w)  # (n_cam, 4, 4)

        return volume_data

    def get_volume_data(self):
        """Get the volume data"""
        return self.volume_data

    def get_radius(self):
        """Return the radius used in render. Return None if not set."""
        if self.get_render_data() is not None:
            return self.get_render_cfgs('radius')
        else:
            return None

    def get_volume_dict(self):
        """Return the volume dict use in volume extracting. Return None if not set."""
        if self.get_volume_data() is not None:
            vol = self.get_volume_data()['Vol']
            volume_dict = {'grid_pts': torch_to_np(vol.get_corner()), 'lines': vol.get_bound_lines()}
            return volume_dict
        else:
            return None

    @torch.no_grad()
    def run_infer(self, model, get_model_feed_in, infer_dir):
        """Run inference for novel view rendering and mesh extraction
           We turn it to eval() mode at beginning but don't turn it to train() mode at the end,
           you need to do it outside this func if needed

           The mesh extraction are all in cpu now. In the future, may need gpu method.

        Args:
            model: model used for inference
            get_model_feed_in: method to prepare input into model
            infer_dir: dir to save the result

        Returns:
            write inference output into eval_dir.
        """
        model.eval()
        files = {'render': None, 'volume': None}

        # render image
        if self.render_data is not None and len(self.render_data['inputs']) > 0:
            files['render'] = self.run_infer_render(self.render_data, get_model_feed_in, model)

        # extract volume/mesh
        if self.volume_data is not None:
            files['volume'] = self.run_infer_volume(self.volume_data, model)

        # not doing anything, return None
        if files['render'] is None and files['volume'] is None:
            files = None

        # write infer files
        self.write_infer_files(files, infer_dir)

        return files

    @torch.no_grad()
    def run_infer_render(self, data, get_model_feed_in, model):
        """Run render inference and return lists of different video frames. Return None if not render is run."""
        volume_render_out = self.run_infer_render_volume(data, get_model_feed_in, model)
        surface_render_out = self.run_infer_render_surface(data, get_model_feed_in, model)

        if len(volume_render_out) == 0 and len(surface_render_out) == 0:
            return None

        output = {
            'volume_out': volume_render_out,
            'surface_out': surface_render_out,
        }

        return output

    def run_infer_render_volume(self, data, get_model_feed_in, model):
        """Get the volume rendering output. Return a list of rgb images. Empty list if input is empty"""
        volume_render_out = []
        for idx, input in enumerate(data['inputs']):
            total_forward_time = 0.0
            self.logger.add_log('Rendering video {}...'.format(idx))
            images = []
            for rays in input:
                feed_in, batch_size = get_model_feed_in(rays, self.device)  # only read rays_o/d here, (1, WH, 3)
                assert batch_size == 1, 'Only one image is sent to model at once for inference...'

                time0 = time.time()
                output = model(feed_in, inference_only=True)
                total_forward_time += (time.time() - time0)

                # get rgb only
                rgb = output['rgb']  # (1, HW, 3)
                rgb = img_to_uint8(torch_to_np(rgb).copy()).reshape(self.H, self.W, 3)  # (H, W, 3), bgr
                images.append(rgb)

            # repeat the image
            images = images * self.get_render_cfgs('repeat')[idx]
            volume_render_out.append(images)

            self.logger.add_log(
                '    Render {} image, each hw({}/{}) total time {:.2f}s'.format(
                    len(input), self.H, self.W, total_forward_time
                )
            )
            self.logger.add_log('    Each image takes time {:.2f}s'.format(total_forward_time / float(len(input))))

        return volume_render_out

    def run_infer_render_surface(self, data, get_model_feed_in, model):
        """Get the surface rendering output. Return a list of rgb images. Empty list if no need to run"""
        surface_render_out = []
        if 'surface_render' in data and data['surface_render'] is not None:
            # reset model chunk rays for faster processing
            origin_chunk_rays = model.get_chunk_rays()
            chunk_rays_factor = 1
            if 'chunk_rays_factor' in data['surface_render']:
                chunk_rays_factor = data['surface_render']['chunk_rays_factor']
            model.set_chunk_rays(origin_chunk_rays * chunk_rays_factor)

            for idx, input in enumerate(data['inputs']):
                total_forward_time = 0.0
                self.logger.add_log('Rendering video {} by surface rendering...'.format(idx))
                images = []
                for rays in input:
                    feed_in, batch_size = get_model_feed_in(rays, self.device)  # only read rays_o/d here, (1, WH, 3)
                    assert batch_size == 1, 'Only one image is sent to model at once for inference...'

                    time0 = time.time()
                    output = model.surface_render(feed_in, **data['surface_render'])  # call surface rendering
                    total_forward_time += (time.time() - time0)

                    # get rgb only
                    rgb = output['rgb']  # (1, HW, 3)
                    rgb = img_to_uint8(torch_to_np(rgb).copy()).reshape(self.H, self.W, 3)  # (H, W, 3), bgr
                    images.append(rgb)

                # repeat the image
                images = images * self.get_render_cfgs('repeat')[idx]
                surface_render_out.append(images)

                self.logger.add_log(
                    '   Surface Render {} image, each hw({}/{}) total time {:.2f}s'.format(
                        len(input), self.H, self.W, total_forward_time
                    )
                )
                self.logger.add_log('    Each image takes time {:.2f}s'.format(total_forward_time / float(len(input))))

            # set back
            model.set_chunk_rays(origin_chunk_rays)

        return surface_render_out

    @torch.no_grad()
    def run_infer_volume(self, data, model, max_pts=200000, max_faces=500000):
        """Run volume inference and return pts and mesh
        Reduce pts and faces num for visual in plotly/plt
        # TODO: extract from sigma/sdf now. we may need to extract from model if model support
        """
        volume_out = {'pc': None, 'mesh': None}
        self.logger.add_log('Extracting volume from model...')
        volume = data['Vol']
        n_grid = self.get_volume_cfgs('n_grid')
        level = self.get_volume_cfgs('level')
        grad_dir = self.get_volume_cfgs('grad_dir')
        # volume for extraction
        volume_pts = volume.get_volume_pts()  # (n_grid^3, 3) pts in torch
        voxel_size = volume.get_voxel_size()  # (3,) tuple
        volume_len = volume.get_len()  # (3,) tuple

        # move to gpu
        if self.device == 'gpu':
            volume_pts = volume_pts.cuda()

        # reset model chunk for faster processing
        origin_chunk_pts = model.get_chunk_pts()
        model.set_chunk_pts(origin_chunk_pts * self.get_volume_cfgs('chunk_pts_factor'))

        # get point cloud output
        volume_out['pc'], sigma, valid_sigma = self.run_infer_volume_point_cloud(
            model, volume_pts, n_grid, level, grad_dir, max_pts
        )

        # get mesh output
        if np.any(valid_sigma):
            volume_out['mesh'] = self.run_infer_volume_mesh(
                model, sigma, level, n_grid, grad_dir, voxel_size, volume_len, max_faces
            )

        # set back in case training
        model.set_chunk_pts(origin_chunk_pts)

        if volume_out['pc'] is None and volume_out['mesh'] is None:
            return None

        return volume_out

    def run_infer_volume_point_cloud(self, model, volume_pts, n_grid, level, grad_dir, max_pts=200000):
        """Extract point cloud from volume field.
           Return point_cloud in full/simplified result as a dict.
           Return None if not run.
        """
        pc_out = None
        # use zeros dir to represent volume pts dir
        time0 = time.time()
        sigma, rgb = model.forward_pts_dir(volume_pts, None)
        sigma, rgb = torch_to_np(sigma), torch_to_np(rgb)
        self.logger.add_log('    Forward {}^3 time for model is {:.2f}s'.format(n_grid, time.time() - time0))
        self.logger.add_log('    Sigma value range {:.2f}-{:.2f}'.format(sigma.min(), sigma.max()))

        # for 3d point cloud visual, valid sigma is area with large enough sigma
        if grad_dir == 'descent':
            valid_sigma = (sigma >= level)  # (n^3,)
        else:
            valid_sigma = (sigma <= level)  # (n^3,)

        if not np.any(valid_sigma):
            self.logger.add_log('You do not have any valid extracting, please check the isolevel', level='warning')
        else:
            pc_out = {}
            valid_pts = torch_to_np(volume_pts)[valid_sigma]  # (n_valid, 3)
            valid_rgb = rgb[valid_sigma]  # (n_valid, 3)
            n_pts = valid_pts.shape[0]
            self.logger.add_log('    Getting {} valid pts'.format(n_pts))
            pc_out['full'] = {'pts': valid_pts.copy(), 'color': valid_rgb.copy()}
            # sample for plot
            if n_pts > max_pts:
                self.logger.add_log('    Sample to {} pts'.format(max_pts))
                choice = np.random.choice(range(n_pts), max_pts, replace=False)
                valid_pts = valid_pts[choice]
                valid_rgb = valid_rgb[choice]
            pc_out['sample'] = {'pts': valid_pts.copy(), 'color': valid_rgb.copy()}

        return pc_out, sigma, valid_sigma

    def run_infer_volume_mesh(self, model, sigma, level, n_grid, grad_dir, voxel_size, volume_len, max_faces=500000):
        """Extract mesh from volume field.
           Return mesh in full/simplified result as a dict.
           Return None if not run.
        """
        mesh_out = None
        sigma = sigma.reshape((n_grid, n_grid, n_grid))  # (n, n, n)

        try:
            # full mesh
            time0 = time.time()
            verts, faces, _ = extract_mesh(sigma.copy(), level, voxel_size, volume_len, grad_dir)
            self.logger.add_log('    Extract mesh time {:.2f}s'.format(time.time() - time0))
            self.logger.add_log('    Extract {} verts, {} faces'.format(verts.shape[0], faces.shape[0]))

            mesh_out = {'full': self.get_mesh_components(verts, faces, model)}

            # simplify mesh if need
            time0 = time.time()
            if faces.shape[0] <= max_faces:  # do not need to run again
                mesh_out['simplify'] = mesh_out['full']
            else:
                verts_sim, faces_sim = simplify_mesh(verts, faces, max_faces)
                self.logger.add_log('    Simplify mesh time {:.2f}s'.format(time.time() - time0))
                self.logger.add_log('    Simplify {} verts, {} faces'.format(verts_sim.shape[0], faces_sim.shape[0]))
                mesh_out['simplify'] = self.get_mesh_components(verts_sim, faces_sim, model)

            # add c2w and intrinsic for mesh rendering
            if 'render' in self.get_render_cfgs():
                mesh_out['render'] = {
                    'type': self.get_render_cfgs('type'),
                    'H': self.H,
                    'W': self.W,
                    'fps': self.get_render_cfgs('fps'),
                    'repeat': self.get_render_cfgs('repeat'),
                    'c2w': self.get_render_data()['c2w'],
                    'intrinsic': self.get_render_data()['intrinsic'],
                    'backend': self.get_render_data()['backend'],
                    'device': next(model.parameters()).device  # to allow gpu usage for rendering
                }

        except ValueError:
            self.logger.add_log('Can not extract mesh from volume', level='warning')

        return mesh_out

    def get_mesh_components(self, verts, faces, model):
        """Get all the mesh components(colors, normals) using model"""
        vert_normals, face_normals = get_normals(verts, faces)
        face_centers = get_face_centers(verts, faces)
        n_verts, n_faces = verts.shape[0], faces.shape[0]

        # get vert_colors, view point is the reverse normal
        vert_view_dir = -vert_normals
        vert_pts = torch.tensor(verts, dtype=self.dtype)  # (n, 3)
        vert_view_dir = torch.tensor(vert_view_dir, dtype=self.dtype)  # (n, 3)

        # move to gpu
        if self.device == 'gpu':
            vert_pts = vert_pts.cuda()
            vert_view_dir = vert_view_dir.cuda()

        time0 = time.time()
        _, vert_colors = model.forward_pts_dir(vert_pts, vert_view_dir)
        vert_colors = torch_to_np(vert_colors)
        self.logger.add_log('    Get verts color for all {} verts takes {:.2f}s'.format(n_verts, time.time() - time0))

        # get face_colors, view point is the reverse normal
        face_view_dir = -face_normals
        face_center_pts = torch.tensor(face_centers, dtype=self.dtype)  # (n, 3)
        face_view_dir = torch.tensor(face_view_dir, dtype=self.dtype)  # (n, 3)

        # move to gpu
        if self.device == 'gpu':
            face_center_pts = face_center_pts.cuda()
            face_view_dir = face_view_dir.cuda()

        time0 = time.time()
        _, face_colors = model.forward_pts_dir(face_center_pts, face_view_dir)
        face_colors = torch_to_np(face_colors)
        self.logger.add_log('    Get faces color for all {} faces takes {:.2f}s'.format(n_faces, time.time() - time0))

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

    def write_infer_files(self, files, folder):
        """Write infer result to the folder. Only show limit number of pts and mesh"""
        if files is None:
            self.logger.add_log('No inference perform...', level='warning')
            return

        # store the geometry and rendering results
        geo_folder = osp.join(folder, 'geometry')
        os.makedirs(geo_folder, exist_ok=True)
        render_folder = osp.join(folder, 'render')
        os.makedirs(render_folder, exist_ok=True)

        # reduce memory for mesh rendering
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # write down video for volume rendering
        if files['render'] is not None and len(files['render']['volume_out']) > 0:
            self.write_volume_render_video(files['render']['volume_out'], render_folder)

        # write down video for surface rendering
        if files['render'] is not None and len(files['render']['surface_out']) > 0:
            self.write_surface_render_video(files['render']['surface_out'], render_folder)

        # write down extract mesh
        if files['volume'] is not None:
            if 'pc' in files['volume'] and files['volume']['pc'] is not None:
                self.write_extract_point_cloud(files['volume']['pc'], geo_folder)

            if 'mesh' in files['volume'] and files['volume']['mesh'] is not None:
                mesh = files['volume']['mesh']
                self.write_extract_mesh(mesh, geo_folder)

                if 'render' in mesh:  # render the mesh only
                    self.write_mesh_render_video(mesh['full'], mesh['render'], render_folder)

    def write_volume_render_video(self, volume_out, render_folder):
        """Write the volume render output"""
        for vid, frames in enumerate(volume_out):
            video_path = osp.join(
                render_folder, 'volume_render_video{}_{}_n{}_fps{}.mp4'.format(
                    vid, self.render_cfgs['type'][vid], self.render_cfgs['n_cam'][vid], self.render_cfgs['fps']
                )
            )
            write_video(frames, video_path, fps=self.render_cfgs['fps'])
        self.logger.add_log('Write volume render videos to {}'.format(render_folder))

    def write_surface_render_video(self, surface_out, render_folder):
        """Write the surface render output"""
        for vid, frames in enumerate(surface_out):
            video_path = osp.join(
                render_folder, 'surface_render_video{}_{}_n{}_fps{}.mp4'.format(
                    vid, self.render_cfgs['type'][vid], self.render_cfgs['n_cam'][vid], self.render_cfgs['fps']
                )
            )
            write_video(frames, video_path, fps=self.render_cfgs['fps'])
        self.logger.add_log('Write surface render videos to {}'.format(render_folder))

    def write_extract_point_cloud(self, pc, geo_folder):
        """Write the full/simplified point cloud extraction"""
        # full pts to ply
        pts = pc['full']['pts']
        pts_colors = pc['full']['color']
        pc_file = osp.join(geo_folder, 'pc_extract.ply')
        save_point_cloud(pc_file, pts, pts_colors)
        # sample pts to plotly
        pts = pc['sample']['pts']
        pts_colors = pc['sample']['color']
        # draw pts in plotly
        file_path = osp.join(geo_folder, 'pc_extract.png')
        draw_3d_components(
            points=pts,
            point_colors=pts_colors,
            point_size=10,
            volume=self.get_volume_dict(),
            title='valid pts({}) from volume'.format(pts.shape[0]),
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )
        self.logger.add_log('Write point cloud visual to {}'.format(geo_folder))

    def write_extract_mesh(self, mesh, geo_folder):
        """Write the full/simplified mesh extraction output"""
        # save full mesh as .ply file, save verts/faces only
        mesh_file = osp.join(geo_folder, 'mesh_extract.ply')
        save_meshes(
            mesh_file, mesh['full']['verts'], mesh['full']['faces'], mesh['full']['vert_colors'],
            mesh['full']['face_colors'], mesh['full']['vert_normals'], mesh['full']['face_normals']
        )
        mesh_geo_file = osp.join(geo_folder, 'mesh_extract_geo.ply')
        save_meshes(mesh_geo_file, mesh['full']['verts'], mesh['full']['faces'], geo_only=True)

        # draw simplified in plotly
        verts_sim, faces_sim = mesh['simplify']['verts'], mesh['simplify']['faces']
        face_colors_sim = mesh['simplify']['face_colors']
        verts_by_faces, _ = get_verts_by_faces(verts_sim, faces_sim, None)
        file_path = osp.join(geo_folder, 'mesh_extract.png')
        draw_3d_components(
            volume=self.get_volume_dict(),
            meshes=[verts_by_faces],
            face_colors=[face_colors_sim],
            title='Meshes ({} faces) extract from volume'.format(verts_by_faces.shape[0]),
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )
        self.logger.add_log('Write mesh visual to {}'.format(geo_folder))

    def write_mesh_render_video(self, mesh, render, render_folder):
        """Write the rasterization ouptut of the mesh given inference camera poses"""
        for path_id, (path_type, c2w) in enumerate(zip(render['type'], render['c2w'])):
            # render color images
            color_imgs = render_mesh_images(
                mesh['verts'],
                mesh['faces'],
                mesh['vert_colors'],
                mesh['face_colors'],
                mesh['vert_normals'],
                mesh['face_normals'],
                render['H'],
                render['W'],
                invert_poses(c2w),
                render['intrinsic'],
                render['backend'],
                single_image_mode=True,
                device=self.device
            )  # (n_cam, h, w, 3)

            file_path = osp.join(render_folder, 'color_mesh_render_{}.mp4'.format(path_type))
            write_video([color_imgs[idx] for idx in range(color_imgs.shape[0])] * render['repeat'][path_id], file_path,
                        True, render['fps'])

            # render geometry only
            geo_imgs = render_mesh_images(
                mesh['verts'],
                mesh['faces'],
                None,
                None,
                mesh['vert_normals'],
                mesh['face_normals'],
                render['H'],
                render['W'],
                invert_poses(c2w),
                render['intrinsic'],
                render['backend'],
                single_image_mode=True,
                device=self.device
            )  # (n_cam, h, w, 3)

            file_path = osp.join(render_folder, 'geo_mesh_render_{}.mp4'.format(path_type))
            write_video([geo_imgs[idx] for idx in range(geo_imgs.shape[0])] * render['repeat'][path_id], file_path,
                        True, render['fps'])

            self.logger.add_log('Write mesh rendering by {} to {}'.format(render['backend'], render_folder))
