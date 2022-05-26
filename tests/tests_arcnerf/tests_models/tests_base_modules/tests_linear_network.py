#!/usr/bin/python
# -*- coding: utf-8 -*-

from itertools import combinations
import os
import os.path as osp
import unittest

import torch

from arcnerf.geometry.mesh import extract_mesh, get_verts_by_faces
from arcnerf.geometry.volume import Volume
from arcnerf.models.base_modules import GeoNet, RadianceNet
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.cfgs_utils import dict_to_obj
from common.utils.torch_utils import torch_to_np, chunk_processing
from common.utils.logger import Logger
from common.visual import get_colors
from tests.tests_arcnerf.tests_models.tests_base_modules import log_base_model_info, RESULT_DIR


class TestDict(unittest.TestCase):
    """This tests the GeoNet and RadianceNet"""

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10

    def tests_geonet(self):
        x = torch.ones((self.batch_size, 3))
        # normal case
        model = GeoNet(input_ch=3)
        y, feat = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        self.assertEqual(feat.shape, (self.batch_size, 256))
        # W_feat <= 0
        model = GeoNet(input_ch=3, W_feat=0)
        y, _ = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        # multi skips
        model = GeoNet(input_ch=3, skips=[1, 2], skip_reduce_output=True)
        y, feat = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        # act
        cfg = {'type': 'softplus', 'beta': 100}
        cfg = dict_to_obj(cfg)
        model = GeoNet(input_ch=3, act_cfg=cfg)
        y, feat = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        self.assertEqual(feat.shape, (self.batch_size, 256))
        # siren
        model = GeoNet(input_ch=3, use_siren=True, skips=[])
        y, feat = model(x)
        self.assertEqual(y.shape, (self.batch_size, 1))
        self.assertEqual(feat.shape, (self.batch_size, 256))
        # forward with normal output and geo value only
        model = GeoNet(input_ch=3)
        geo_value, feat, grad = model.forward_with_grad(x)
        self.assertEqual(x.shape, grad.shape)
        self.assertEqual(feat.shape, (self.batch_size, 256))
        geo_value = model.forward_geo_value(x)
        self.assertEqual(geo_value.shape, (self.batch_size, ))

    def tests_geonet_detail(self):
        model = GeoNet()
        logger = Logger(path=osp.join(RESULT_DIR, 'geonet.txt'), keep_console=False)
        n_pts = 4096 * 128
        feed_in = torch.ones((n_pts, 3))
        log_base_model_info(logger, model, feed_in, n_pts)

    def tests_geonet_geo_siren_init(self):
        GEOINIT_DIR = osp.abspath(osp.join(RESULT_DIR, 'geo_init'))
        os.makedirs(GEOINIT_DIR, exist_ok=True)

        # params
        radius_init = 0.5  # for inner obj radius
        side = 1.5  # for volume
        n_grid = 64
        chunk_pts = 65536

        use_siren_choice = [True, False]
        for use_siren in use_siren_choice:
            model = GeoNet(
                input_ch=3,
                D=1,
                skips=[] if use_siren else [4],
                geometric_init=True,
                radius_init=radius_init,
                weight_norm=True,
                skip_reduce_output=True,
                norm_skip=True,
                use_siren=use_siren,
                act_cfg=dict_to_obj({'type': 'softplus'}),
                W_feat=256
            )

            gpu_on_func = False
            if torch.cuda.is_available():
                model.cuda()
                gpu_on_func = True  # move volume pts to gpu

            model.pretrain_siren()

            # check 3 pts
            sur_xyz = ((radius_init**2) / 3)**0.5
            pts = torch.tensor([
                [0.0, 0.0, 0.0],  # (inner center, sdf < 0)
                [sur_xyz] * 3,  # (on surface, sdf = 0)
                [2 * radius_init] * 3
            ])  # (outer space, sdf > 0)
            if torch.cuda.is_available():
                pts = pts.cuda()
            geo_value = model(pts)[0]
            self.assertTrue(geo_value[0] < 0 and geo_value[-1] > 0 and abs(geo_value[1]) < 0.8)

            # volume and get pts
            volume = Volume(n_grid=n_grid, side=side)
            voxel_size = volume.get_voxel_size()
            volume_len = volume.get_len()
            volume_pts = volume.get_volume_pts()  # (n_grid^3, 3) pts in torch
            volume_dict = {
                'grid_pts': torch_to_np(volume.get_corner()),
                'lines': volume.get_bound_lines(),
                'faces': volume.get_bound_faces()
            }

            geo_value = chunk_processing(model, chunk_pts, gpu_on_func, volume_pts)[0][:, 0]  # (n_grid^3, 1) sdf value
            valid_sigma = (geo_value <= 0)  # inside pts (n^3,)
            valid_pts = torch_to_np(volume_pts[valid_sigma])  # (n_valid, 3)
            geo_value = torch_to_np(geo_value).reshape((n_grid, n_grid, n_grid))

            # draw pts in plotly
            file_path = osp.join(GEOINIT_DIR, 'model_geo_init_pc(siren).png' if use_siren else 'model_geo_init_pc.png')
            draw_3d_components(
                points=valid_pts,
                point_colors=get_colors('blue', to_np=True),
                volume=volume_dict,
                title='init pts with geo value < 0',
                save_path=file_path,
                plotly=True,
                plotly_html=True
            )

            # draw the mesh in plotly
            verts, faces, _ = extract_mesh(geo_value.copy(), 0, voxel_size, volume_len, grad_dir='ascent')
            verts_by_faces, _ = get_verts_by_faces(verts, faces)
            # draw mesh in plotly
            file_path = osp.join(
                GEOINIT_DIR, 'model_geo_init_mesh(siren).png' if use_siren else 'model_geo_init_mesh.png'
            )
            draw_3d_components(
                volume=volume_dict,
                meshes=[verts_by_faces],
                mesh_colors=get_colors('blue', to_np=True),
                title='init mesh with inner geo value < 0',
                save_path=file_path,
                plotly=True,
                plotly_html=True
            )

    def tests_radiancenet(self):
        xyz = torch.rand((self.batch_size, 3))
        view_dirs = torch.rand((self.batch_size, 3))
        normals = torch.rand((self.batch_size, 3))
        feat = torch.rand((self.batch_size, 256))
        modes = ['p', 'v', 'n', 'f']
        modes = sum([list(map(list, combinations(modes, i))) for i in range(len(modes) + 1)], [])
        for mode in modes:
            if len(mode) == 0:
                continue
            mode = ''.join(mode)
            model = RadianceNet(mode=mode, W=128, D=8, W_feat_in=256)
            y = model(xyz, view_dirs, normals, feat)
            self.assertEqual(y.shape, (self.batch_size, 3))
            self.assertTrue(torch.all(torch.logical_and(y >= 0, y <= 1)))


if __name__ == '__main__':
    unittest.main()
