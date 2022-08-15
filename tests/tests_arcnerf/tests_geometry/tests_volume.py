#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np
import torch

from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from arcnerf.geometry.volume import Volume
from arcnerf.render.ray_helper import get_zvals_from_near_far
from arcnerf.visual.plot_3d import draw_3d_components
from common.utils.torch_utils import torch_to_np, np_wrapper
from common.visual import get_combine_colors, get_colors

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results', 'volume'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.side = 1.5
        cls.n_grid = 4
        cls.xyz_len = [1.0, 2.0, 1.5]
        cls.radius = 2.0
        cls.origin = (0.5, 0.5, 0)

    def tests_center_volume(self):
        volume = Volume(n_grid=self.n_grid, side=self.side)
        self.assertEqual(torch_to_np(volume.get_origin()).shape, (3, ))
        self.assertEqual(torch_to_np(volume.get_corner()).shape, (8, 3))
        self.assertEqual(torch_to_np(volume.get_grid_pts()).shape, ((self.n_grid + 1)**3, 3))

        volume_dict = {
            'grid_pts': torch_to_np(volume.get_corner()),  # (8, 3)
            'lines': volume.get_bound_lines(),  # (2*6, 3)
            'faces': volume.get_bound_faces()  # (6, 4, 3)
        }
        file_path = osp.join(RESULT_DIR, 'center_volume_bound.png')
        draw_3d_components(
            volume=volume_dict,
            sphere_radius=self.radius,
            title='center volume with bound lines/faces',
            save_path=file_path
        )

        volume_dict = {
            'grid_pts': torch_to_np(volume.get_grid_pts()),  # (n+1^3, 3)
            'volume_pts': torch_to_np(volume.get_volume_pts()),  # (n^3, 3)
            'lines': volume.get_dense_lines(),  # 3(n+1)^3 * (2, 3)
            'faces': volume.get_dense_faces()  # (3(n+1)n^2, 4, 3)
        }
        file_path = osp.join(RESULT_DIR, 'center_volume_dense.png')
        draw_3d_components(
            volume=volume_dict,
            sphere_radius=self.radius,
            title='center volume with dense lines/faces',
            save_path=file_path
        )

    def tests_custom_volume(self):
        volume = Volume(n_grid=self.n_grid)
        volume.set_params(self.origin, None, self.xyz_len)

        volume_dict = {
            'grid_pts': torch_to_np(volume.get_corner()),  # (8, 3)
            'lines': volume.get_bound_lines(),  # (2*6, 3)
            'faces': volume.get_bound_faces()  # (3(n+1)n^2, 4, 3)
        }
        file_path = osp.join(RESULT_DIR, 'custom_volume_bound.png')
        draw_3d_components(
            volume=volume_dict,
            sphere_radius=self.radius,
            title='custom volume with bound lines/faces',
            save_path=file_path
        )

        volume_dict = {
            'grid_pts': torch_to_np(volume.get_grid_pts()),  # (n+1^3, 3)
            'volume_pts': torch_to_np(volume.get_volume_pts()),  # (n^3, 3)
            'lines': volume.get_dense_lines(),  # 3(n+1)^3 * (2, 3)
            'faces': volume.get_dense_faces()  # (3(n+1)n^2, 4, 3)
        }
        file_path = osp.join(RESULT_DIR, 'custom_volume_dense.png')
        draw_3d_components(
            volume=volume_dict,
            sphere_radius=self.radius,
            title='custom volume with dense lines/faces',
            save_path=file_path
        )

    def tests_ray_volume(self):
        """Test some ray volume interaction function"""
        volume = Volume(n_grid=self.n_grid, side=self.side)
        rays_o = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        rays_d = np.array([[-1.5, -1.0, -0.8]], dtype=np.float32)
        rays_d = normalize(rays_d)

        n_pts = 15
        zvals = np.linspace(0.0, 4.0, n_pts, dtype=np.float32)[None]  # (1, n_pts)
        pts = np_wrapper(get_ray_points_by_zvals, rays_o, rays_d, zvals).reshape(-1, 3)  # (n_pts, 3)
        pts_in_boundary = np_wrapper(volume.check_pts_in_grid_boundary, pts, volume.get_corner())  # (n_in, )

        # assign color to pts
        pts_colors = get_combine_colors(['red'], [pts.shape[0]])
        green_color = get_colors('green', to_int=False, to_np=True)
        pts_colors[pts_in_boundary] = green_color

        # get voxel idx
        voxel_idx, valid_idx = np_wrapper(volume.get_voxel_idx_from_xyz, pts)
        self.assertTrue(np.all(np.equal(valid_idx, pts_in_boundary)))

        # get valid grid_pts
        grid_pts_valid = np_wrapper(volume.get_grid_pts_by_voxel_idx, voxel_idx[valid_idx])  # (n_in, 8, 3)
        grid_pts_weights_valid = np_wrapper(volume.cal_weights_to_grid_pts, pts[valid_idx], grid_pts_valid)  # (n_in, 8)
        self.assertTrue(np.all(grid_pts_weights_valid > 0))

        # check by direct method
        voxel_grid = np_wrapper(volume.get_voxel_grid_info_from_xyz, pts)
        self.assertTrue(np.allclose(voxel_grid[-1], grid_pts_weights_valid))

        # get lines and draw lines with weights as width
        lines = []
        line_widths = []
        line_colors = []
        for idx in range(grid_pts_valid.shape[0]):
            for grid_pts_idx in range(grid_pts_valid.shape[1]):
                line = np.concatenate([pts[valid_idx][idx][None], grid_pts_valid[idx][grid_pts_idx][None]])
                lines.append(line)
                weight = grid_pts_weights_valid[idx][grid_pts_idx]
                line_widths.append(weight * 10.0)
                line_color = get_colors('red', to_int=False, to_np=True)[None]
                line_colors.append(line_color)
        line_colors = np.concatenate(line_colors, axis=0)

        # draw
        volume_dict = {
            'grid_pts': torch_to_np(volume.get_grid_pts()),  # (n+1^3, 3)
            'lines': volume.get_dense_lines(),  # 3(n+1)^3 * (2, 3)
        }

        file_path = osp.join(RESULT_DIR, 'volume_pts_interpolation.png')
        draw_3d_components(
            points=pts,
            point_colors=pts_colors,
            lines=lines,
            line_widths=line_widths,
            line_colors=line_colors,
            volume=volume_dict,
            title='volume with ray pts interpolation',
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

    def tests_get_collect_grid_pts_by_voxel_idx(self):
        volume = Volume(n_grid=512, side=self.side)
        batch_size = 4096
        pts = torch.rand((batch_size, 3))

        # get voxel idx
        voxel_idx, valid_idx = np_wrapper(volume.get_voxel_idx_from_xyz, pts)

        # collect by voxel_idx
        grid_pts_1 = np_wrapper(volume.collect_grid_pts_by_voxel_idx, voxel_idx[valid_idx])

        # get by voxel_idx
        grid_pts_2 = np_wrapper(volume.get_grid_pts_by_voxel_idx, voxel_idx[valid_idx])

        self.assertTrue(np.allclose(grid_pts_1, grid_pts_2))

    def tests_voxel_bitfield(self):
        n_grid = 4
        volume = Volume(n_grid=n_grid, side=self.side)
        volume.set_up_voxel_bitfield()
        self.assertEqual(volume.get_voxel_bitfield().shape, (n_grid, n_grid, n_grid))
        self.assertEqual(volume.get_n_occupied_voxel(), n_grid**3)

        # reset
        volume.reset_voxel_bitfield(False)
        self.assertEqual(volume.get_n_occupied_voxel(), 0)
        volume.reset_voxel_bitfield(True)
        self.assertEqual(volume.get_n_occupied_voxel(), n_grid**3)

        # random occ
        occupancy = torch.rand((n_grid, n_grid, n_grid))
        occupancy = (occupancy > 0.5).type(torch.bool)
        volume.update_bitfield(occupancy)
        self.assertEqual(volume.get_n_occupied_voxel(), occupancy.sum())

        volume_dict = {
            'grid_pts': torch_to_np(volume.get_occupied_grid_pts().view(-1, 3)),  # (8N, 3)
            'lines': volume.get_occupied_lines(),  # (12N) * (2, 3)
            'faces': volume.get_occupied_faces()  # (6N, 4, 3)
        }

        file_path = osp.join(RESULT_DIR, 'voxel_occupancy.png')
        draw_3d_components(
            volume=volume_dict,
            title='volume with occupancy indicator',
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

    def tests_ray_voxel_pass_through(self):
        volume = Volume(n_grid=8, side=self.side)
        volume.set_up_voxel_bitfield()
        n_rays = 8
        rays_o = np.random.rand(n_rays, 3) * 2.0
        rays_d = -normalize(rays_o + np.random.rand(n_rays, 3))  # point to origin

        # get the voxel_idx that rays pass through
        occ = np_wrapper(volume.get_ray_pass_through, rays_o, rays_d)
        np_wrapper(volume.update_bitfield, occ)

        volume_dict = {
            'lines': volume.get_occupied_lines(),  # (12N) * (2, 3)
            'faces': volume.get_occupied_faces()  # (6N, 4, 3)
        }

        file_path = osp.join(RESULT_DIR, 'ray_pass_through_volume.png')
        draw_3d_components(
            rays=(rays_o, rays_d * 4.0),
            volume=volume_dict,
            title='ray pass though volume selection',
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

    def tests_get_voxel_grid_pts_by_voxel_idx(self):
        volume = Volume(n_grid=4, side=self.side)  # (-0.75, 0.75)
        volume.set_up_voxel_bitfield()
        batch_size = 32
        pts = torch.rand((batch_size, 3)) * 2.0 - 1.0  # (-1, 1)
        pts_colors = get_combine_colors(['blue'], [batch_size])

        # get unique voxel idx
        voxel_idx, valid_idx = np_wrapper(volume.get_voxel_idx_from_xyz, pts)
        uni_valid_voxel_idx = np_wrapper(volume.get_unique_voxel_idx, voxel_idx[valid_idx])

        # update occupancy
        volume.reset_voxel_bitfield(False)
        np_wrapper(volume.update_bitfield_by_voxel_idx, voxel_idx[valid_idx])
        self.assertEqual(volume.get_n_occupied_voxel(), uni_valid_voxel_idx.shape[0])

        # get grid pts and voxel pts
        grid_pts = np_wrapper(volume.get_occupied_grid_pts).reshape(-1, 3)  # (8N, 3)
        voxel_pts = np_wrapper(volume.get_occupied_voxel_pts)  # (N, 3)

        volume_dict = {
            'grid_pts': grid_pts,  # (8N, 3)
            'volume_pts': voxel_pts,  # (N, 3)
            'lines': volume.get_occupied_lines(),  # (12N) * (2, 3)
            'faces': volume.get_occupied_faces()  # (6N, 4, 3)
        }

        file_path = osp.join(RESULT_DIR, 'voxel_grid_pts.png')
        draw_3d_components(
            points=torch_to_np(pts),
            point_colors=pts_colors,
            volume=volume_dict,
            title='occupied voxels with grid_pts and voxel_pts',
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

    def tests_ray_voxel_intersection(self):
        n_grid = 8
        offset = 2
        volume = Volume(n_grid=n_grid, side=self.side)
        volume.set_up_voxel_bitfield()
        n_rays = 8
        rays_o = np.random.rand(n_rays, 3) * 2.0
        rays_d = -normalize(rays_o + 2.0 * np.random.rand(n_rays, 3))  # point to origin

        # remove the outside voxels
        occ = torch.zeros((n_grid, n_grid, n_grid)).type(torch.bool)
        center_len = n_grid - 2 * offset
        occ_center = torch.ones((center_len, center_len, center_len)).type(torch.bool)
        occ[offset:n_grid - offset, offset:n_grid - offset, offset:n_grid - offset] = occ_center
        np_wrapper(volume.update_bitfield, occ)

        # make a coarse volume
        empty_ratio = 0.8
        voxel_idx = volume.get_occupied_voxel_idx()
        n_occ = voxel_idx.shape[0]
        empty_voxel_idx = torch.randperm(n_occ)[:int(n_occ * empty_ratio)]
        empty_voxel_idx = voxel_idx[empty_voxel_idx]
        volume.update_bitfield_by_voxel_idx(empty_voxel_idx, occ=False)

        volume_dict = {
            'lines': volume.get_occupied_lines(),  # (12N) * (2, 3)
            'faces': volume.get_occupied_faces()  # (6N, 4, 3)
        }

        # on a full volume
        lines = volume.get_lines_from_vertices(volume.get_corner(), 2)
        line_colors = get_colors('red', to_int=False, to_np=True)
        near, far, _, mask = np_wrapper(volume.ray_volume_intersection, rays_o, rays_d)
        mask = mask[:, 0]
        n_no_hit = np.sum(~mask)
        near, far = near[mask].reshape(-1, 1), far[mask].reshape(-1, 1)
        zvals = np_wrapper(get_zvals_from_near_far, near, far, 8)
        pts = np_wrapper(get_ray_points_by_zvals, rays_o[mask], rays_d[mask], zvals).reshape(-1, 3)
        ray_colors = get_combine_colors(['red'], [n_rays])
        ray_colors[mask] = get_colors('blue', to_int=False, to_np=True)

        file_path = osp.join(RESULT_DIR, 'ray_voxel_full_volume.png')
        draw_3d_components(
            points=pts,
            lines=lines,
            line_colors=line_colors,
            rays=(rays_o, rays_d * 4.0),
            ray_colors=ray_colors,
            volume=volume_dict,
            title='ray volume intersection on full volume, {}/{} rays no hit'.format(n_no_hit, n_rays),
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

        # on dense voxels
        near, far, _, mask = np_wrapper(volume.ray_volume_intersection_in_occ_voxel, rays_o, rays_d)
        mask = mask[:, 0]
        n_no_hit = np.sum(~mask)
        near, far = near[mask].reshape(-1, 1), far[mask].reshape(-1, 1)
        zvals = np_wrapper(get_zvals_from_near_far, near, far, 8)
        pts = np_wrapper(get_ray_points_by_zvals, rays_o[mask], rays_d[mask], zvals).reshape(-1, 3)
        ray_colors = get_combine_colors(['red'], [n_rays])
        ray_colors[mask] = get_colors('blue', to_int=False, to_np=True)

        file_path = osp.join(RESULT_DIR, 'ray_voxel_dense_voxels.png')
        draw_3d_components(
            points=pts,
            lines=lines,
            line_colors=line_colors,
            rays=(rays_o, rays_d * 4.0),
            ray_colors=ray_colors,
            volume=volume_dict,
            title='ray volume intersection on dense voxels, {}/{} rays no hit'.format(n_no_hit, n_rays),
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )

        # on dense voxels bounding volume
        _, _, _, mask_dense = np_wrapper(volume.ray_volume_intersection_in_occ_voxel, rays_o, rays_d, False)
        n_no_hit_dense = np.sum(~mask_dense)
        near, far, _, mask = np_wrapper(volume.ray_volume_intersection_in_occ_voxel, rays_o, rays_d, True)
        mask = mask[:, 0]
        n_no_hit = np.sum(~mask)
        near, far = near[mask].reshape(-1, 1), far[mask].reshape(-1, 1)
        zvals = np_wrapper(get_zvals_from_near_far, near, far, 8)
        pts = np_wrapper(get_ray_points_by_zvals, rays_o[mask], rays_d[mask], zvals).reshape(-1, 3)
        ray_colors = get_combine_colors(['red'], [n_rays])
        ray_colors[mask] = get_colors('blue', to_int=False, to_np=True)

        lines = volume.get_lines_from_vertices(volume.get_corner(), 2)
        line_colors = get_colors('red', to_int=False, to_np=True)
        lines_bounding = volume.get_lines_from_vertices(volume.get_occupied_bounding_corner(), 2)
        lines.extend(lines_bounding)

        file_path = osp.join(RESULT_DIR, 'ray_voxel_dense_voxels_bounding.png')
        title = 'ray volume intersection on bounding volume, '
        title += '{}/{} rays no hit, '.format(n_no_hit, n_rays)
        title += '{}/{} rays not hit dense but on bounding'.format(n_no_hit_dense - n_no_hit, n_rays)
        draw_3d_components(
            points=pts,
            lines=lines,
            line_colors=line_colors,
            rays=(rays_o, rays_d * 4.0),
            ray_colors=ray_colors,
            volume=volume_dict,
            title=title,
            save_path=file_path,
            plotly=True,
            plotly_html=True
        )
