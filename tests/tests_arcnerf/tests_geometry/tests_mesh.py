#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import os
import os.path as osp
import unittest

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
from arcnerf.geometry.poses import generate_cam_pose_on_sphere, invert_poses
from arcnerf.geometry.volume import Volume
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.torch_utils import torch_to_np
from common.utils.video_utils import write_video
from common.visual import get_colors_from_cm, get_combine_colors

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results', 'mesh'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.side = 1.5
        cls.n_grid = 256
        cls.radius = 0.5
        cls.hourglass_h = 1.0
        cls.level = 0
        cls.max_faces = 500  # 500000 is okay for mesh with no rays

    def make_sphere_sdf(self, volume_pts):
        """make the sphere sdf with radius from volume_pts. Inside -/outside +"""
        sdf = self.radius - np.linalg.norm(volume_pts, axis=-1)
        sdf *= -1.0

        return sdf

    def make_hourglass_sdf(self, volume_pts, n_grid):
        """make the hourglass sdf with radius from volume_pts Inside -/outside +"""
        sdf = abs(volume_pts[:, 1])**2 - (volume_pts[:, 0]**2 + volume_pts[:, 2]**2)
        sdf = sdf.reshape((n_grid, n_grid, n_grid))
        offset = int(n_grid * (1 - (self.hourglass_h / self.side)) / 2)
        sdf[:, :offset, :] = -1.0
        sdf[:, -offset:, :] = -1.0
        sdf = sdf.reshape(-1) * -1.0

        return sdf

    def get_mesh_components(self, verts, faces):
        """Get all components of mesh from verts and faces"""
        n_verts, n_faces = verts.shape[0], faces.shape[0]
        vert_normals, face_normals = get_normals(verts, faces)
        self.assertEqual(vert_normals.shape, (n_verts, 3))
        self.assertEqual(face_normals.shape, (n_faces, 3))
        face_centers = get_face_centers(verts, faces)
        self.assertEqual(face_centers.shape, (n_faces, 3))

        # random color
        vert_colors = get_colors_from_cm(1024, to_np=True)
        vert_colors = vert_colors.repeat(max(1, math.ceil(float(n_verts) / 1024.0)), 0)[:n_verts]  # (n_v, 3)
        face_colors = get_colors_from_cm(1024, to_np=True)
        face_colors = face_colors.repeat(max(1, math.ceil(float(n_faces) / 1024.0)), 0)[:n_faces]  # (n_f, 3)
        self.assertEqual(vert_colors.shape, (n_verts, 3))
        self.assertEqual(face_colors.shape, (n_faces, 3))

        return face_centers, vert_normals, face_normals, vert_colors, face_colors

    def run_mesh(self, type='sphere', grad_dir='descent'):
        """Simulate a object with sigma >0 inside, <0 outside. Extract voxels"""
        assert type in ['sphere', 'hourglass'], 'Invalid object'
        object_dir = osp.join(RESULT_DIR, type)
        os.makedirs(object_dir, exist_ok=True)

        volume = Volume(n_grid=self.n_grid, side=self.side)
        volume_pts = torch_to_np(volume.get_volume_pts())  # (n^3, 3)
        volume_size = volume.get_volume_size()
        volume_len = volume.get_len()
        volume_dict = {
            'grid_pts': torch_to_np(volume.get_corner()),
            'lines': volume.get_bound_lines(),
            'faces': volume.get_bound_faces()
        }

        # simulate a object sdf
        if type == 'sphere':
            sdf = self.make_sphere_sdf(volume_pts)  # (n^3, )
        else:
            sdf = self.make_hourglass_sdf(volume_pts, self.n_grid)  # (n^3, )

        # reverse sdf to density
        geo_title = 'sdf'
        if grad_dir == 'descent':
            sdf *= -1  # inside gets +
            geo_title = 'sigma'

        object_dir = osp.join(object_dir, geo_title)
        os.makedirs(object_dir, exist_ok=True)

        # get full mesh
        sdf = sdf.reshape((self.n_grid, self.n_grid, self.n_grid))
        verts, faces, vert_normals_ = extract_mesh(sdf.copy(), self.level, volume_size, volume_len, grad_dir)
        face_centers, vert_normals, face_normals, vert_colors, face_colors = self.get_mesh_components(verts, faces)

        # save ply
        mesh_file = osp.join(object_dir, 'full_mesh_ngrid{}.ply'.format(self.n_grid))
        save_meshes(mesh_file, verts, faces, vert_colors, face_colors, vert_normals, face_normals)

        mesh_geo_file = osp.join(object_dir, 'full_mesh_geo_ngrid{}.ply'.format(self.n_grid))
        save_meshes(mesh_geo_file, verts, faces, None, None, vert_normals, face_normals, geo_only=True)

        # only in gpu, cpu too long
        if torch.cuda.is_available():
            # pytorch3d rendering full mesh
            try:  # may not have pytorch3d
                color_img = self.render_mesh(
                    verts, faces, vert_colors, face_colors, vert_normals, face_normals, 'pytorch3d'
                )
                file_path = osp.join(object_dir, 'pytorch3d_color_render.mp4')
                write_video([color_img[idx] for idx in range(color_img.shape[0])], file_path, True)

                geo_img = self.render_mesh(verts, faces, None, None, vert_normals, face_normals, 'pytorch3d')
                file_path = osp.join(object_dir, 'pytorch3d_geo_render.mp4')
                write_video([geo_img[idx] for idx in range(geo_img.shape[0])], file_path, True)

                sil_img = self.render_mesh(verts, faces, None, None, vert_normals, face_normals, 'pytorch3d', True)
                file_path = osp.join(object_dir, 'pytorch3d_sil_render.mp4')
                write_video([sil_img[idx] for idx in range(sil_img.shape[0])], file_path, True)
            except ImportError:
                pass

            # open3d rendering, only geometry.
            try:  # may not have open3d
                geo_img = self.render_mesh(verts, faces, None, None, vert_normals, face_normals, 'open3d')
                file_path = osp.join(object_dir, 'open3d_geo_render.mp4')
                write_video([geo_img[idx] for idx in range(geo_img.shape[0])], file_path, True)
            except ImportError:
                pass

        # get simplified mesh for plotly 3d. Otherwise too large to save and open
        verts, faces = simplify_mesh(verts, faces, self.max_faces)
        self.assertLessEqual(faces.shape[0], self.max_faces)
        face_centers, vert_normals, face_normals, vert_colors, face_colors = self.get_mesh_components(verts, faces)
        n_verts, n_faces = verts.shape[0], faces.shape[0]
        ray_colors = get_combine_colors(['blue', 'green'], [n_verts, n_faces])
        rays = (
            np.concatenate([verts, face_centers], axis=0), np.concatenate([vert_normals, face_normals], axis=0) / 20.0
        )  # factor
        self.assertEqual(rays[0].shape, (n_verts + n_faces, 3))
        self.assertEqual(rays[1].shape, (n_verts + n_faces, 3))
        # get verts and rays for 3d visual
        verts_by_faces, mean_face_colors = get_verts_by_faces(verts, faces, vert_colors)
        self.assertEqual(verts_by_faces.shape, (n_faces, 3, 3))
        self.assertEqual(mean_face_colors.shape, (n_faces, 3))

        # draw mesh in plotly
        file_path = osp.join(object_dir, 'simplify_mesh_ngrid{}.png'.format(self.n_grid))
        draw_3d_components(
            volume=volume_dict,
            rays=rays,
            ray_colors=ray_colors,
            meshes=[verts_by_faces],
            face_colors=[face_colors],
            title='Meshes extract from volume v{}/f{} by - {}'.format(n_verts, n_faces, geo_title),
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

    def render_mesh(self, verts, faces, vert_colors, face_colors, vert_normals, face_normals, backend, sil_mode=False):
        n_cam = 30
        c2w = generate_cam_pose_on_sphere('circle', self.side * 2.0, n_cam)  # (n_cam, 4, 4)
        w2c = invert_poses(c2w)
        h, w = 300, 400
        focal = 500.0
        intrinsic = np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0, 0, 1]])  # (3, 3)

        verts = torch.tensor(verts, dtype=torch.float32).cuda()
        device = verts.device

        # set up renderer
        img_list = render_mesh_images(
            verts,
            faces,
            vert_colors,
            face_colors,
            vert_normals,
            face_normals,
            h,
            w,
            w2c,
            intrinsic,
            backend=backend,
            device=device,
            sil_mode=sil_mode
        )

        return img_list

    def tests_mesh_sphere_sdf(self):
        self.run_mesh('sphere', grad_dir='ascent')

    def tests_mesh_hourglass_sdf(self):
        self.run_mesh('hourglass', grad_dir='ascent')

    def tests_mesh_sphere_sigma(self):
        self.run_mesh('sphere')

    def tests_mesh_hourglass_sigma(self):
        self.run_mesh('hourglass')
