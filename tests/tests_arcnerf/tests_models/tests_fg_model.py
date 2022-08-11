#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import torch

from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from arcnerf.models.base_modules.linear_network_module import BaseGeoNet
from arcnerf.models.fg_model import FgModel
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.cfgs_utils import dict_to_obj
from common.utils.torch_utils import torch_to_np

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestModelDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_rays = 32
        cls.bounding_radius = 3.0
        cls.base_cfgs = {
            'model': {
                'chunk_rays': 4096,
                'chunk_pts': 4096 * 32,
                'rays': {
                    'bounding_radius': cls.bounding_radius
                }
            }
        }
        cls.result_dir = osp.join(RESULT_DIR, 'fg_model')
        os.makedirs(cls.result_dir, exist_ok=True)

    @staticmethod
    def to_cuda(item):
        """Move model or tensor to cuda"""
        if torch.cuda.is_available():
            item = item.cuda()

        return item

    def create_feed_in_to_cuda(self):
        rays_o = torch.rand(self.n_rays, 3) * 3.0
        rays_d = -normalize(rays_o + torch.rand(self.n_rays, 3) * 2.0)  # point to origin
        feed_in = {
            'rays_o': rays_o,
            'rays_d': rays_d,
        }

        for k, v in feed_in.items():
            feed_in[k] = self.to_cuda(v)

        return feed_in

    def get_zvals_np_from_model(self, inputs, model):
        # call get_near_far_from_rays
        near, far = model.get_near_far_from_rays(inputs)
        self.assertEqual(near.shape, (self.n_rays, 1))
        self.assertEqual(far.shape, (self.n_rays, 1))
        # draw the sampling pts
        zvals = model.get_zvals_from_near_far(near, far, 16)
        pts = get_ray_points_by_zvals(inputs['rays_o'], inputs['rays_d'], zvals)
        pts = torch_to_np(pts).reshape(-1, 3)

        near, far = torch_to_np(near), torch_to_np(far)

        return near, far, pts

    def run_fg_model_tests(self, model, type='none'):
        inputs = self.create_feed_in_to_cuda()
        model = self.to_cuda(model)
        rays_o = torch_to_np(inputs['rays_o'])
        rays_d = torch_to_np(inputs['rays_d'])

        near, far, pts = self.get_zvals_np_from_model(inputs, model)

        # sphere or volume
        volume_dict = None
        if model.get_obj_bound_and_type()[1] == 'volume':
            volume = model.get_obj_bound_and_type()[0]
            volume_dict = {
                'grid_pts': torch_to_np(volume.get_corner()),
                'lines': volume.get_bound_lines(),
                'faces': volume.get_bound_faces()
            }

        radius = [self.bounding_radius]
        if model.get_obj_bound_and_type()[1] == 'sphere':
            radius_obj = model.get_obj_bound_and_type()[0].get_radius(in_float=True)
            radius.append(radius_obj)

        file_path = osp.join(self.result_dir, 'struct_{}_sampling_pts.png'.format(type))
        draw_3d_components(
            points=pts,
            rays=(rays_o, far * rays_d),
            sphere_radius=radius,
            volume=volume_dict,
            title='Sampling pts by inner obj bound - {}'.format(type),
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

    def run_volume_prunning_acc(self, model):
        inputs = self.create_feed_in_to_cuda()
        model = self.to_cuda(model)
        rays_o = torch_to_np(inputs['rays_o'])
        rays_d = torch_to_np(inputs['rays_d'])

        # pruning the voxels
        model.set_factor(100.0)
        model.optim_obj_bound(10)  # warmup

        volume = model.get_obj_bound_and_type()[0]
        volume_dict = {
            'grid_pts': torch_to_np(volume.get_occupied_grid_pts().view(-1, 3)),
            'lines': volume.get_occupied_lines(),
            'faces': volume.get_occupied_faces()
        }
        file_path = osp.join(self.result_dir, 'pruned_volume_coarse.png')
        draw_3d_components(
            volume=volume_dict,
            sphere_radius=1.0,  # inner radius
            title='Pruned volume coarse stage',
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

        # this could update by a refined one
        model.set_factor(1.0)
        model.optim_obj_bound(500)  # not warmup

        volume = model.get_obj_bound_and_type()[0]
        volume_dict = {
            'grid_pts': torch_to_np(volume.get_occupied_grid_pts().view(-1, 3)),
            'lines': volume.get_occupied_lines(),
            'faces': volume.get_occupied_faces()
        }
        file_path = osp.join(self.result_dir, 'pruned_volume_fine.png')
        draw_3d_components(
            volume=volume_dict,
            sphere_radius=1.0,  # inner radius
            title='Pruned volume fine stage',
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

        # pruning without acc sampling
        near, far, pts = self.get_zvals_np_from_model(inputs, model)
        volume = model.get_obj_bound_and_type()[0]
        volume_dict = {
            'grid_pts': torch_to_np(volume.get_occupied_grid_pts().view(-1, 3)),
            'lines': volume.get_occupied_lines(),
            'faces': volume.get_occupied_faces()
        }
        file_path = osp.join(self.result_dir, 'pruned_volume_no_acc_sample.png')
        draw_3d_components(
            points=pts,
            rays=(rays_o, far * rays_d),
            volume=volume_dict,
            title='Pruned volume without acc sampling',
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

        # pruning with acc sampling
        model.ray_sample_acc = True
        near, far, pts = self.get_zvals_np_from_model(inputs, model)
        volume = model.get_obj_bound_and_type()[0]
        volume_dict = {
            'grid_pts': torch_to_np(volume.get_occupied_grid_pts().view(-1, 3)),
            'lines': volume.get_occupied_lines(),
            'faces': volume.get_occupied_faces()
        }
        file_path = osp.join(self.result_dir, 'pruned_volume_with_acc_sample.png')
        draw_3d_components(
            points=pts,
            rays=(rays_o, far * rays_d),
            volume=volume_dict,
            title='Pruned volume with acc sampling',
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

    def tests_set_up_volume_bound_model(self):
        volume_model_cfgs = self.base_cfgs.copy()
        volume_cfgs = {'volume': {'n_grid': 4, 'side': 2.0}}
        volume_model_cfgs['model']['obj_bound'] = volume_cfgs
        fg_model = FakeFGModel(dict_to_obj(volume_model_cfgs))
        self.run_fg_model_tests(fg_model, 'volume')

        # with pruning
        volume_model_cfgs['model']['epoch_optim'] = 16
        volume_model_cfgs['model']['epoch_optim_warmup'] = 256
        volume_model_cfgs['model']['ema_optim_decay'] = 0.0  # for updata compare
        volume_model_cfgs['model']['ray_sample_acc'] = False
        fg_model = FakeFGModel(dict_to_obj(volume_model_cfgs))
        self.run_volume_prunning_acc(fg_model)

    def tests_set_up_sphere_bound_model(self):
        sphere_model_cfgs = self.base_cfgs.copy()
        sphere_cfgs = {'sphere': {'radius': 1.0}}
        sphere_model_cfgs['model']['obj_bound'] = sphere_cfgs
        fg_model = FgModel(dict_to_obj(sphere_model_cfgs))
        self.run_fg_model_tests(fg_model, 'sphere')

    def tests_set_up_no_bound_model(self):
        fg_model = FakeFGModel(dict_to_obj(self.base_cfgs))
        self.run_fg_model_tests(fg_model)


class FakeFGModel(FgModel):
    """A synthetic FG model that density is the inverse distance to origin"""

    def __init__(self, cfgs):
        super(FakeFGModel, self).__init__(cfgs)
        self.geo_net = FakeGeoNet(radius=1.0)
        self.radiance_net = None

    def set_factor(self, fac):
        self.geo_net.set_factor(fac)


class FakeGeoNet(BaseGeoNet):
    """A synthetic geonet for density creation"""

    def __init__(self, radius):
        super(FakeGeoNet, self).__init__()
        self.radius = radius
        self.factor = 50.0  # adjust the density value

    def forward(self, x: torch.Tensor):
        dist = self.radius - torch.norm(x, dim=-1, keepdim=True)  # inside is positive
        dist = dist.clamp_min(0.0) * self.factor  # (B, 1)

        return dist, None

    def set_factor(self, fac):
        self.factor = fac
