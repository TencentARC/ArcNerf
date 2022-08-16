# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from .ray import aabb_ray_intersection, get_ray_points_by_zvals
from common.utils.torch_utils import torch_to_np


class Volume(nn.Module):
    """A volume with customized operation"""

    def __init__(
        self, n_grid=None, origin=(0, 0, 0), side=None, xyz_len=None, dtype=torch.float32, requires_grad=False
    ):
        """
        Args:
            n_grid: N of volume/line seg on each side. Each side is divided into n_grid seg with n_grid+1 pts.
                    total num of volume is n_grid**3, total num of grid_pts is (n_grid+1)**3.
                    If n_grid is None, only set the out-bounding lines/pts. By default None.
            origin: origin point(centroid of cube), a tuple of 3
            side: each side len, if None, use xyz_len. If exist, use side only
            xyz_len: len of xyz dim, if None use side
            dtype: dtype of params. By default is torch.float32
            requires_grad: whether the parameters requires grad. If True, waste memory for graphic.
        """
        super(Volume, self).__init__()
        self.requires_grad = requires_grad
        self.dtype = dtype

        # set nn params
        self.n_grid = n_grid
        self.origin = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], dtype=dtype), requires_grad=self.requires_grad)
        self.xyz_len = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], dtype=dtype), requires_grad=self.requires_grad)

        # whether init bitfield
        self.contains_bitfield = False

        # set real value
        if origin is not None and (side is not None or xyz_len is not None):
            self.set_params(origin, side, xyz_len)

    def set_params(self, origin, side, xyz_len):
        """you can call outside to reset the params"""
        assert side is not None or xyz_len is not None, 'Specify at least side or xyz_len'
        self.set_origin(origin)
        self.set_len(side, xyz_len)
        self.set_pts()

    def set_pts(self):
        """Set the pts(range, corner, grid, volume) using the internal origin/lengths"""
        self.cal_range()
        self.cal_corner()
        if self.n_grid is not None:
            self.cal_grid_pts()
            self.cal_volume_pts()

    def set_n_grid(self, n_grid, reset_pts=True):
        """Change the n_grid manually and reset the grid_pts"""
        self.n_grid = n_grid

        # reset grid_pts and volume_pts, the cost here could be large for dense volume
        if reset_pts:
            self.cal_grid_pts()
            self.cal_volume_pts()

    def get_n_grid(self):
        """Get the num of grid"""
        return self.n_grid

    def get_n_voxel(self):
        """Get total num of voxel. Assume n_grid exists."""
        return self.n_grid**3

    def get_n_grid_pts(self):
        """Get total num of grid_pts. Assume n_grid exists."""
        return (self.n_grid + 1)**3

    def get_device(self):
        """Get the device for parameter"""
        return self.origin.device

    @torch.no_grad()
    def set_len(self, side, xyz_len):
        """Set len of each dim"""
        if side is not None:
            self.xyz_len[0] = side
            self.xyz_len[1] = side
            self.xyz_len[2] = side
        else:
            self.xyz_len[0] = xyz_len[0]
            self.xyz_len[1] = xyz_len[1]
            self.xyz_len[2] = xyz_len[2]

    def get_len(self):
        """Return len of each dim, in tuple of float num"""
        return float(self.xyz_len[0]), float(self.xyz_len[1]), float(self.xyz_len[2])

    def expand_len(self, factor):
        """Expand the length of each dim. When requires_grad, do not call this"""
        self.xyz_len.data = self.xyz_len.data * factor
        self.set_pts()

    @torch.no_grad()
    def set_origin(self, origin=(0.0, 0.0, 0.0)):
        """Set the origin """
        self.origin[0] = origin[0]
        self.origin[1] = origin[1]
        self.origin[2] = origin[2]

    def get_origin(self):
        """Gets origin in tensor(3, )"""
        return self.origin

    def cal_range(self):
        """Cal the xyz range(min, max) from origin and sides. range is (3, 2) tensor"""
        xyz_min = self.origin - self.xyz_len / 2.0
        xyz_max = self.origin + self.xyz_len / 2.0
        self.register_buffer('range', torch.cat([xyz_min[:, None], xyz_max[:, None]], dim=-1))

    def get_range(self):
        """Get the xyz range in (3, 2)"""
        return self.range

    def get_diag_len(self):
        """Get the len of diagonal"""
        return float(torch.sqrt(((self.range[:, 1] - self.range[:, 0])**2).sum()))

    def get_full_voxel_idx(self, flatten=False):
        """Get the full voxel_idx

        Args:
            flatten: whether to get flatten idx, by default False
        Returns:
            voxel_idx: If flatten, in (n_grid**3, 3), else in (n_grid, n_grid, n_grid, 3), each value in (0, n_grid)
        """
        x = torch.linspace(0, self.n_grid - 1, self.n_grid, dtype=torch.long, device=self.get_device())  # (n)
        y = torch.linspace(0, self.n_grid - 1, self.n_grid, dtype=torch.long, device=self.get_device())  # (n)
        z = torch.linspace(0, self.n_grid - 1, self.n_grid, dtype=torch.long, device=self.get_device())  # (n)

        voxel_idx = torch.stack(torch.meshgrid(x, y, z), -1)  # (n, n, n, 3)
        if flatten:
            voxel_idx = voxel_idx.view(-1, 3)

        return voxel_idx

    @staticmethod
    def get_eight_permutation(x, y, z):
        """Get the eight permutation from (x1, x2)-(y1, y2)-(z1, z2)

        Args:
            x: (2, 1) tensor, can be index or real value
            y: (2, 1) tensor, can be index or real value
            z:(2, 1) tensor, can be index or real value

        Returns:
            (8, 3) tensor, order is (x1, y1, z1)
                                    (x1, y2, z1)
                                    (x2, y1, z1)
                                    (x2, y2, z1)
                                    (x1, y1, z2)
                                    (x1, y2, z2)
                                    (x2, y1, z2)
                                    (x2, y2, z2)
        """
        x_r = x.unsqueeze(1).repeat(1, 2, 1)  # (2, 2, 1)
        y_r = y.unsqueeze(0).repeat(2, 1, 1)  # (2, 2, 1)
        xy_r = torch.cat([x_r, y_r], -1).view(-1, 2)  # (4, 2)
        xy_r = xy_r.unsqueeze(0).repeat(2, 1, 1)  # (2, 4, 2)
        z_r = z.unsqueeze(1).repeat(1, 4, 1)  # (2, 4, 1)
        xyz_r = torch.cat([xy_r, z_r], -1).view(-1, 3)  # (8, 3)

        return xyz_r

    def get_eight_permutation_index(self):
        """Get the eight permutation from (0, 1)-(0, 1)-(0, 1)

        Returns:
            permute_index: (8, 3) tensor in torch.long
        """
        x_index = torch.tensor([0, 1], dtype=torch.long)[:, None]
        y_index = torch.tensor([0, 1], dtype=torch.long)[:, None]
        z_index = torch.tensor([0, 1], dtype=torch.long)[:, None]
        permute_index = self.get_eight_permutation(x_index, y_index, z_index)  # (8, 3)

        return permute_index

    def cal_corner(self):
        """Cal the eight corner pts given origin and sides. corner is (8, 3) tensor """
        x, y, z = self.range[0][:, None], self.range[1][:, None], self.range[2][:, None]  # (2, 1)
        self.register_buffer('corner', self.get_eight_permutation(x, y, z))

    def get_corner(self, in_grid=False):
        """Get the eight corner. tensor (8, 3). If in_grid, return (2, 2, 2, 3) """
        if in_grid:
            return self.corner.view(2, 2, 2, 3)
        else:
            return self.corner

    def cal_grid_pts(self):
        """Cal all the grid pts. tensor ((self.n_grid+1)^3, 3) """
        x = torch.linspace(float(self.range[0, 0]), float(self.range[0, 1]), self.n_grid + 1)  # (n+1)
        y = torch.linspace(float(self.range[1, 0]), float(self.range[1, 1]), self.n_grid + 1)  # (n+1)
        z = torch.linspace(float(self.range[2, 0]), float(self.range[2, 1]), self.n_grid + 1)  # (n+1)

        grid_pts = torch.stack(torch.meshgrid(x, y, z), -1)  # (n+1, n+1, n+1, 3)
        self.register_buffer('grid_pts', grid_pts.view(-1, 3))

    def get_grid_pts(self, in_grid=False):
        """Get ((self.n_grid+1)^3, 3) grid pts. If in_grid, return (n_grid+1, n_grid+1, n_grid+1, 3) """
        if in_grid:
            return self.grid_pts.view(self.n_grid + 1, self.n_grid + 1, self.n_grid + 1, 3)
        else:
            return self.grid_pts

    def cal_volume_pts(self):
        """Cal all the volume center pts. tensor ((self.n_grid)^3, 3) """
        v_x, v_y, v_z = self.get_voxel_size()  # volume size
        x = torch.linspace(float(self.range[0, 0]) + 0.5 * v_x, float(self.range[0, 1]) - 0.5 * v_x, self.n_grid)  # (n)
        y = torch.linspace(float(self.range[1, 0]) + 0.5 * v_y, float(self.range[1, 1]) - 0.5 * v_y, self.n_grid)  # (n)
        z = torch.linspace(float(self.range[2, 0]) + 0.5 * v_z, float(self.range[2, 1]) - 0.5 * v_z, self.n_grid)  # (n)
        volume_pts = torch.stack(torch.meshgrid(x, y, z), -1)  # (n, n, n, 3)
        self.register_buffer('volume_pts', volume_pts.view(-1, 3))  # (n^3, 3)

    def get_volume_pts(self, in_grid=False):
        """Get ((self.n_grid)^3, 3) volume pts. If in_grid, return (n_grid, n_grid, n_grid, 3) """
        if in_grid:
            return self.volume_pts.view(self.n_grid, self.n_grid, self.n_grid, 3)
        else:
            return self.volume_pts

    def get_voxel_size(self, to_list=True):
        """Get each voxel size on each side. Good for marching cube

        Args:
            to_list: if True, return a tuple of output (x_s, y_s, z_s). Else return a tensor of (3, ). By default False.
        """
        xyz_s = (self.get_range()[:, 1] - self.get_range()[:, 0]) / self.n_grid  # (3, )

        if to_list:
            x_s, y_s, z_s = float(xyz_s[0]), float(xyz_s[1]), float(xyz_s[2])
            return x_s, y_s, z_s
        else:
            return xyz_s

    def get_subdivide_grid_pts(self, volume_range):
        """Get the eight subdivided volume grid_pts by voxel index
                    order should be (x1, y1, z1)
                                    (x1, y2, z1)
                                    (x2, y1, z1)
                                    (x2, y2, z1)
                                    (x1, y1, z2)
                                    (x1, y2, z2)
                                    (x2, y1, z2)
                                    (x2, y2, z2)

        Args:
            volume_range: list of 3, containing (idx_start, idx_end) for xyz

        Returns:
            sub_grid_pts: (8, 8, 3) grid_pts of subdivided volumes
            sub_volume_range: list of 8(sub volume), each is list of 3, containing (idx_start, idx_end) for xyz
        """
        x_start, x_end = volume_range[0]
        y_start, y_end = volume_range[1]
        z_start, z_end = volume_range[2]
        if x_start == x_end - 1 or y_start == y_end - 1 or z_start == z_end - 1:
            raise RuntimeError('Can not be subdivided any more...')

        x_range = [x_start, (x_start + x_end) // 2, x_end]
        y_range = [y_start, (y_start + y_end) // 2, y_end]
        z_range = [z_start, (z_start + z_end) // 2, z_end]

        sub_grid_pts = []
        sub_volume_range = []

        permute_index = self.get_eight_permutation_index()  # (8, 3)
        grid_pts = self.get_grid_pts(True)  # (n_grid+1, n_grid+1, n_grid+1, 3)
        for i in range(permute_index.shape[0]):
            _x, _y, _z = permute_index[i][0], permute_index[i][1], permute_index[i][2]  # 0 and 1
            sub_x_start, sub_x_end = x_range[_x], x_range[_x + 1]
            sub_y_start, sub_y_end = y_range[_y], y_range[_y + 1]
            sub_z_start, sub_z_end = z_range[_z], z_range[_z + 1]
            sub_volume_range.append([(sub_x_start, sub_x_end), (sub_y_start, sub_y_end), (sub_z_start, sub_z_end)])

            grid_pts_list = [
                grid_pts[sub_x_start, sub_y_start, sub_z_start, :].unsqueeze(0),
                grid_pts[sub_x_start, sub_y_end, sub_z_start, :].unsqueeze(0),
                grid_pts[sub_x_end, sub_y_start, sub_z_start, :].unsqueeze(0),
                grid_pts[sub_x_end, sub_y_end, sub_z_start, :].unsqueeze(0),
                grid_pts[sub_x_start, sub_y_start, sub_z_end, :].unsqueeze(0),
                grid_pts[sub_x_start, sub_y_end, sub_z_end, :].unsqueeze(0),
                grid_pts[sub_x_end, sub_y_start, sub_z_end, :].unsqueeze(0),
                grid_pts[sub_x_end, sub_y_end, sub_z_end, :].unsqueeze(0),
            ]
            sub_grid_pts.append(torch.cat(grid_pts_list, dim=0).unsqueeze(0))  # (1, 8, 3)

        sub_grid_pts = torch.cat(sub_grid_pts, dim=0)  # (8, 8, 3)

        return sub_grid_pts, sub_volume_range

    @staticmethod
    def check_pts_in_grid_boundary(pts: torch.Tensor, grid_pts: torch.Tensor):
        """Check whether pts in grid pts boundary. Remind that float accuracy affect the real choice.

        Args:
            pts: (B, 3), points to be check
            grid_pts: (B, 8, 3) or (8, 3), grid pts bounding for each pts. Can be a single set in (8, 3)

        Returns:
            pts_in_boundary: (B, ) whether each pts is in boundary
        """
        n_pts = pts.shape[0]

        if len(grid_pts.shape) == 2:
            grid_pts_expand = torch.repeat_interleave(grid_pts.unsqueeze(0), n_pts, 0)
        else:
            grid_pts_expand = grid_pts

        assert grid_pts_expand.shape[0] == n_pts, 'Invalid num of grid_pts, should be in (B, 8, 3) or (8, 3)'

        # check in min_max boundary
        pts_min_req = (pts >= grid_pts_expand.min(dim=1)[0]).clone().detach().type(torch.bool).to(pts.device)
        pts_max_req = (pts < grid_pts_expand.max(dim=1)[0]).clone().detach().type(torch.bool).to(pts.device)
        pts_in_boundary = torch.logical_and(pts_min_req, pts_max_req)  # (B, 3)
        pts_in_boundary = torch.all(pts_in_boundary, dim=-1)  # (B, )

        return pts_in_boundary

    def get_voxel_idx_from_xyz(self, pts: torch.Tensor):
        """Get the voxel idx from xyz pts. Directly calculate the offset index in each dim.

        Args:
            pts: xyz position of points. (B, 3) tensor of xyz

        Returns:
            voxel_idx: return the voxel index in (B, 3) of each xyz index, range in (0, n_grid).
                       if not in any voxel, return (-1, -1, -1) for this pts. torch.long
            valid_idx: (B, ) mask that pts are in the volume. torch.BoolTensor
        """
        dtype = pts.dtype

        voxel_size = self.get_voxel_size(to_list=False).type(dtype)  # (3,)
        start_point = self.get_range()[:, 0]  # min xyz (3,)
        voxel_idx = (pts - start_point) / voxel_size  # (B, 3)

        valid_idx = torch.logical_and(
            torch.all(voxel_idx >= 0, dim=1), torch.all(voxel_idx < float(self.n_grid), dim=1)
        )

        voxel_idx[~valid_idx] = -1
        voxel_idx = torch.floor(voxel_idx).type(torch.long)

        return voxel_idx, valid_idx

    @staticmethod
    def get_unique_voxel_idx(voxel_idx: torch.Tensor):
        """Get the unique voxel idx by removing the duplicated rows

        Args:
            voxel_idx: voxel index in (B, 3) of each xyz index, range in (0, n_grid).

        Returns:
            uni_voxel_idx: unique voxel_idx in (B_uni, 3)
        """
        return torch.unique(voxel_idx, dim=0)

    def get_grid_pts_idx_by_voxel_idx(self, voxel_idx: torch.Tensor, flatten=True):
        """Get the grid pts index from voxel idx

        Args:
            voxel_idx: (B, 3) tensor of xyz index, should be in (0, n_grid)
            flatten: If flatten, return the flatten index in (B, 8), else in (B, 8, 3). By default True

        Returns:
            grid_pts_idx_by_idx: select grid_pts index by voxel_idx, each index of grid
                                 (B, 8) if flatten, else (B, 8, 3)
        """
        assert 0 <= voxel_idx.min() <= voxel_idx.max() < self.n_grid, 'Voxel idx out of boundary'

        permute_index = self.get_eight_permutation_index().to(voxel_idx.device)  # (8, 3)

        # add 0,1 offset to voxel_idx as grid_pts index
        grid_pts_idx = voxel_idx.unsqueeze(1) + permute_index.unsqueeze(0)  # (B, 8, 3)

        if flatten:
            grid_pts_idx = self.convert_xyz_index_to_flatten_index(grid_pts_idx.view(-1, 3), self.n_grid + 1)  # (B*8, )
            grid_pts_idx = grid_pts_idx.view(-1, 8)  # (B, 8)

        return grid_pts_idx

    def collect_grid_pts_by_voxel_idx(self, voxel_idx: torch.Tensor):
        """Get the grid pts xyz from voxel idx. Collect from full grid_pts is slower than just cal by input

        Args:
            voxel_idx: (B, 3) tensor of xyz index, should be in (0, n_grid)

        Returns:
            grid_pts_by_voxel_idx: select grid_pts xyz values by voxel_idx, (B, 8, 3), each pts of grid
        """
        grid_pts = self.get_grid_pts().to(voxel_idx.device)  # (n+1^3, 3)
        grid_pts_idx = self.get_grid_pts_idx_by_voxel_idx(voxel_idx)  # (B, 8)

        # select by idx
        grid_pts_by_voxel_idx = self.collect_grid_pts_values(grid_pts, grid_pts_idx)  # (B, 8, 3)

        return grid_pts_by_voxel_idx

    def get_grid_pts_by_voxel_idx(self, voxel_idx: torch.Tensor):
        """Get the grid pts xyz from voxel idx, directly calculated from voxel_idx

        Args:
            voxel_idx: (B, 3) tensor of xyz index, should be in (0, n_grid)

        Returns:
            grid_pts_by_voxel_idx: select grid_pts xyz values by voxel_idx, (B, 8, 3), each pts of grid
        """
        grid_pts_idx = self.get_grid_pts_idx_by_voxel_idx(voxel_idx, flatten=False)  # (B, 8, 3)

        # base position(xyz_min) and voxel size
        voxel_size = self.get_voxel_size(to_list=False)  # (3,)
        start_pos = self.get_range()[:, 0]  # (3,)

        grid_pts_by_voxel_idx = grid_pts_idx * voxel_size + start_pos

        return grid_pts_by_voxel_idx

    def get_voxel_pts_by_voxel_idx(self, voxel_idx: torch.Tensor):
        """Get the voxel center pts(part of volume_pts) xyz from voxel idx, directly calculated from voxel_ids

        Args:
            voxel_idx: (B, 3) tensor of xyz index, should be in (0, n_grid)

        Returns:
            voxel_pts_by_voxel_idx: select voxel center xyz values by voxel_idx, (B, 3), each pts of voxel
        """
        # base position(xyz_min) and voxel size
        voxel_size = self.get_voxel_size(to_list=False)  # (3,)
        start_pos = self.get_range()[:, 0]  # (3,)

        voxel_pts_by_voxel_ids = voxel_idx * voxel_size + 0.5 * voxel_size + start_pos

        return voxel_pts_by_voxel_ids

    def cal_weights_to_grid_pts(self, pts: torch.Tensor, grid_pts: torch.Tensor):
        """Calculate the weights of each grid_pts to pts by trilinear interpolation

        Args:
            pts: (B, 3) pts, each point should be in the 8 pts grid.
            grid_pts: (B, 8, 3), eight corner pts. Will use the first and last one.
                       order is (x1, y1, z1)
                                (x1, y2, z1)
                                (x2, y1, z1)
                                (x2, y2, z1)
                                (x1, y1, z2)
                                (x1, y2, z2)
                                (x2, y1, z2)
                                (x2, y2, z2)

        Returns:
            weights: weights to each grid pts in (B, 8) in (0~1). Order corresponding to grid_pts order.
        """
        n_pts = pts.shape[0]
        assert grid_pts.shape == (n_pts, 8, 3), 'Shape not match'

        w_xyz = (pts - grid_pts[:, 0, :]) / (grid_pts[:, -1, :] - grid_pts[:, 0, :])  # (B, 3)
        w_xyz = w_xyz.clip(0.0, 1.0)  # in case some pts out of grid

        # linear interpolation
        permute_index = self.get_eight_permutation_index()  # (8, 3)
        weights = (permute_index[:, 0] * w_xyz[:, 0:1] + (1 - permute_index[:, 0]) * (1 - w_xyz[:, 0:1])) * \
                  (permute_index[:, 1] * w_xyz[:, 1:2] + (1 - permute_index[:, 1]) * (1 - w_xyz[:, 1:2])) * \
                  (permute_index[:, 2] * w_xyz[:, 2:3] + (1 - permute_index[:, 2]) * (1 - w_xyz[:, 2:3]))  # (B, 8)

        return weights

    def get_voxel_grid_info_from_xyz(self, pts: torch.Tensor):
        """Get the voxel and grid pts info(index, pos) directly from xyz position

        Args:
            pts: xyz position of points. (B, 3) tensor of xyz

        Returns:
            voxel_idx: return the voxel index in (B, 3) of each xyz index, range in (0, n_grid).
                       if not in any voxel, return (-1, -1, -1) for this pts. torch.long
            valid_idx: (B, ) mask that pts are in the volume. torch.BoolTensor
            grid_pts_idx: (B_valid, 8), valid grid_pts index. Return None if no valid.
            grid_pts: (B_valid, 8), valid grid_pts xyz pos. Return None if no valid.
            grid_pts_weights: (B_valid, 8), weights on valid grid_pts. Return None if no valid.
        """
        device = pts.device

        # get base info
        voxel_size = self.get_voxel_size(to_list=False)  # (3,)
        start_point = self.get_range()[:, 0]  # min xyz (3,)
        permute_index = self.get_eight_permutation_index().to(device)  # (8, 3)

        # get voxel and valid info
        voxel_idx = (pts - start_point) / voxel_size  # (B, 3)
        valid_idx = torch.logical_and(
            torch.all(voxel_idx >= 0, dim=1), torch.all(voxel_idx < float(self.n_grid), dim=1)
        )
        voxel_idx[~valid_idx] = -1
        voxel_idx = torch.floor(voxel_idx).type(torch.long)

        grid_pts_idx, grid_pts, grid_pts_weights = None, None, None
        if torch.any(valid_idx):
            # for all valid voxel, get grid_pts index and position
            grid_pts_idx = voxel_idx[valid_idx].unsqueeze(1) + permute_index.unsqueeze(0)  # (B_valid, 8, 3)
            grid_pts = grid_pts_idx * voxel_size + start_point[0]  # (B_valid, 8, 3)

            # get_weight for valid pts
            w_xyz = (pts[valid_idx] - grid_pts[:, 0, :]) / voxel_size  # (B_valid, 3)
            w_xyz = w_xyz.clip(0.0, 1.0)  # in case some pts out of grid

            grid_pts_weights = \
                (permute_index[:, 0] * w_xyz[:, 0:1] + (1 - permute_index[:, 0]) * (1 - w_xyz[:, 0:1])) * \
                (permute_index[:, 1] * w_xyz[:, 1:2] + (1 - permute_index[:, 1]) * (1 - w_xyz[:, 1:2])) * \
                (permute_index[:, 2] * w_xyz[:, 2:3] + (1 - permute_index[:, 2]) * (1 - w_xyz[:, 2:3]))  # (B_valid, 8)

        return voxel_idx, valid_idx, grid_pts_idx, grid_pts, grid_pts_weights

    def interpolate(self, values: torch.Tensor, weights: torch.Tensor, voxel_idx: torch.Tensor):
        """Interpolate values by getting value on grid_pts in each voxel and multiply weights
        You should assume the values are on the same device.

        Args:
            values:  (n_grid+1^3, ...) values
            weights: (B, 8), grid weights on each grid pts in voxels, weights in each voxel add up to 1.0.
            voxel_idx: (B, 3) tensor of xyz index, should be in (0, n_grid)

        Returns:
            values_by_weights: (B, ...) interpolated values by weights of grid_pts in voxel
        """
        grid_pts_idx = self.get_grid_pts_idx_by_voxel_idx(voxel_idx)  # (B, 8)

        # select by idx
        values_on_grid_pts = self.collect_grid_pts_values(values, grid_pts_idx)  # (B, 8, ...)

        # multiply weights
        values_by_weights = self.interpolate_values_by_weights(values_on_grid_pts, weights)

        return values_by_weights

    @staticmethod
    def interpolate_values_by_weights(values: torch.Tensor, weights: torch.Tensor):
        """Interpolate the values collect from grid_pts by weights

        Args:
            values:  (B, 8, ...) values
            weights: (B, 8), grid weights on each grid pts in voxels, weights in each voxel add up to 1.0.

        Returns:
            values_by_weights: (B, ...) interpolated values by weights
        """
        weights_expand = weights.view(*weights.shape, *(1, ) * (len(values.shape) - 2))  # expand like values
        values_by_weights = values * weights_expand  # (B, 8, ...)
        values_by_weights = values_by_weights.sum(1)  # (B, ...)

        return values_by_weights

    @staticmethod
    def convert_xyz_index_to_flatten_index(xyz: torch.Tensor, n):
        """Convert xyz index to flatten index

        Args:
            xyz: (B, 3) index in torch.long
            n: num of offset on each dim. Generally use n_grid or n_grid + 1

        Returns:
            flatten_index: (B, ) flatten index
        """
        flatten_index = xyz[:, 0] * (n**2) + xyz[:, 1] * n + xyz[:, 2]  # (B, )

        return flatten_index

    @staticmethod
    def convert_flatten_index_to_xyz_index(flatten_index: torch.Tensor, n):
        """Convert flatten index to xyz index

        Args:
            flatten_index: (B, ) flatten index in torch.long
            n: num of offset on each dim. Generally use n_grid or n_grid + 1

        Returns:
            xyz: (B, 3) index in torch.long
        """
        xyz_index = torch.zeros((flatten_index.shape[0], 3), dtype=torch.long, device=flatten_index.device)

        for i in range(3):
            xyz_index[:, 2 - i] = flatten_index % n
            flatten_index = torch.div(flatten_index, n, rounding_mode='trunc')

        return xyz_index

    def collect_grid_pts_values(self, values: torch.Tensor, grid_pts_idx: torch.Tensor):
        """Collect values on grid_pts(xyz, feature, value) by grid_pts_idx

        Args:
            values: (n_grid+1^3, ...) values, all the grid_pts_values. Only the first shape is required.
            grid_pts_idx: (B, 8), grid pts idx of each sample

        Returns:
            values_by_grid_pts_idx: (B, 8, ...), selected values on each grid_pts,
                                    shape[2:] same as values.shape[1:]
        """
        assert values.shape[0] == self.get_n_grid_pts(), 'Invalid dim of values...'

        # select by idx
        values_by_grid_pts_idx = values[grid_pts_idx, ...]  # (B, 8, ...)

        return values_by_grid_pts_idx

    def ray_volume_intersection(self, rays_o: torch.Tensor, rays_d: torch.Tensor, in_occ_voxel=False):
        """Calculate the rays intersection with the out-bounding surfaces

        Args:
            rays_o: ray origin, (N_rays, 3)
            rays_d: ray direction, (N_rays, 3)
            in_occ_voxel: If True, only calculate intersection between rays and current occ rays.
                          It can reduced calculation. When some voxels are masked not occupied. By default False.

        Returns:
            near: near intersection zvals. (N_rays, 1)
                  If only 1 intersection: if not tangent, same as far; else 0. clip by 0.
            far:  far intersection zvals. (N_rays, 1)
                  If only 1 intersection: if not tangent, same as far; else 0.
            pts: (N_rays, 2, 3), each ray has near/far two points with each volume.
                                      if nan, means no intersection at this ray
            mask: (N_rays, 1), show whether each ray has intersection with the volume, BoolTensor
        """
        if in_occ_voxel:  # in occupied sample
            near, far, pts, mask = self.ray_volume_intersection_in_occ_voxel(rays_o, rays_d)
        else:  # full volume
            aabb_range = self.get_range()[None].to(rays_o.device)  # (1, 3, 2)
            near, far, pts, mask = aabb_ray_intersection(rays_o, rays_d, aabb_range)
            pts = pts[:, 0, :, :]  # (N_rays, 1, 2, 3) -> (N_rays, 2, 3)

        return near, far, pts, mask

    def ray_volume_intersection_in_occ_voxel(self, rays_o: torch.Tensor, rays_d: torch.Tensor, force=False):
        """Ray volume intersection in occupied voxels only.

        If the n_rays * n_volume is large, hard to calculate on all the volumes
        (eg. 4096 rays * [128**3] volume, float32, near/far takes = 32GB memory each, can not afford.
        It will calculate the result on smallest remaining bounding voxels.
        We only allow at most (n_rays * n_volume < 4096 * [32**3]) dense calculation, which takes 0.5GB-near/far
        for this calculation, and it takes time even on GPU.

        Args:
            rays_o: ray origin, (N_rays, 3)
            rays_d: ray direction, (N_rays, 3)
            force: If True, only calculate in the smallest bounding volume, instead of every voxels. By default False.
                    But this is not accurate for every rays. Some rays not hit the dense voxel may be included.

        Returns:
            see returns in `ray_volume_intersection`
        """
        assert self.bitfield is not None, 'You must have the occupancy indicator, You should not use dense voxels.'

        n_rays = rays_o.shape[0]
        n_occ = self.get_n_occupied_voxel()
        max_allow = 4096 * 32**3
        if force or n_rays * n_occ > max_allow:  # find the smallest bounding of remaining voxels
            aabb_range = self.get_occupied_bounding_range()[None]  # (1, 3, 2)
            near, far, pts, mask = aabb_ray_intersection(rays_o, rays_d, aabb_range)
            pts = pts[:, 0, :, :]  # (N_rays, 1, 2, 3) -> (N_rays, 2, 3)
        else:  # use all remaining voxels
            grid_pts = self.get_occupied_grid_pts()  # (n_occ, 3)
            aabb_range = torch.cat([grid_pts[:, 0, :].unsqueeze(-1), grid_pts[:, -1, :].unsqueeze(-1)], dim=-1)
            near, far, _, mask = aabb_ray_intersection(rays_o, rays_d, aabb_range)
            # find the rays near/far with any hit voxels
            near[~mask], far[~mask] = 1e6, -1e6
            near = torch.min(near, dim=1)[0][:, None]  # (n_rays, 1)
            far = torch.max(far, dim=1)[0][:, None]  # (n_rays, 1)
            mask = torch.any(mask, dim=1, keepdim=True)  # (n_rays, 1)

            # those not hit rays get all 0, recalculate pts
            near[~mask], far[~mask] = 0.0, 0.0
            pts = torch.cat(
                [get_ray_points_by_zvals(rays_o, rays_d, near),
                 get_ray_points_by_zvals(rays_o, rays_d, far)], dim=1
            )  # (n_rays, 2, 3)

        return near, far, pts, mask

    def get_ray_pass_through(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, return_voxel_idx=False, in_occ_voxel=False
    ):
        """Get voxel_idx that rays pass through. It takes all n**3 voxel for aabb intersection, will take time.
        Do not use when n_rays * n_occ_voxel is large.

        Args:
            rays_o: ray origin, (N_rays, 3)
            rays_d: ray direction, (N_rays, 3)
            return_voxel_idx: If True, return the voxel idx of those passed through voxels. by default False
            in_occ_voxel: If True, only calculate intersection between rays and current occ rays.
                          It can reduced calculation. When some voxels are masked not occupied. By default False.
                          You are only allowed to return the voxel_idx in this case

        Returns:
            out: (n_grid, n_grid, n_grid) tensor of xyz index that rays pass through
                 If return_voxel_idx, return a voxel_idx in (N, 3).Return None if no intersection.
        """
        if in_occ_voxel and self.bitfield is not None:
            assert return_voxel_idx, 'If fine the ray pass through from occupied voxels, you can only return voxel_idx'
            grid_pts = self.get_occupied_grid_pts()  # (n_occ, 3)
        else:
            voxel_idx = self.get_full_voxel_idx(flatten=True)  # (n_grid**3, 3)
            grid_pts = self.get_grid_pts_by_voxel_idx(voxel_idx)  # (n_grid**3, 8, 3)

        aabb_range = torch.cat([grid_pts[:, 0, :].unsqueeze(-1), grid_pts[:, -1, :].unsqueeze(-1)], dim=-1)  # (n, 3, 2)

        _, _, _, mask = aabb_ray_intersection(rays_o, rays_d, aabb_range)  # (n_rays, n_grid**3)
        mask = torch.any(mask, dim=0)  # (n_grid**3) / (n)

        if not return_voxel_idx:
            return mask.view(self.n_grid, self.n_grid, self.n_grid)

        # return voxel_idx
        if torch.sum(mask) == 0:  # all rays not hit
            return None

        voxel_idx = torch.where(mask)[0].type(torch.long)
        voxel_idx = self.convert_flatten_index_to_xyz_index(voxel_idx, self.n_grid)

        return voxel_idx

    def set_up_voxel_bitfield(self, init_occ=True):
        """Set up the bitfield as a torch.bool tensor in shape (n_grid, n_grid, n_grid), each value is the occupancy
        Bitfield representation in (n_grid^3 // 8, uint8) can save memory, but torch is not flexible in bit manipulation
        You need customized cuda func for that.

        Args:
            init_occ: init occupancy, by default True(all voxels occupied).

        Returns:
            bitfield: a (n_grid, n_grid, n_grid) bool tensor
        """
        assert self.n_grid is not None, 'Voxel bitfield must be set for known resolution volume'
        self.contains_bitfield = True

        if init_occ:
            bitfield = torch.ones((self.n_grid, self.n_grid, self.n_grid), dtype=torch.bool)
        else:
            bitfield = torch.zeros((self.n_grid, self.n_grid, self.n_grid), dtype=torch.bool)
        self.register_buffer('bitfield', bitfield)

    def get_voxel_bitfield(self, flatten=False):
        """Get the voxel bitfield. Return None if bitfield no init.

        Args:
            flatten: whether to flatten the voxel idx

        Returns:
            bitfield: if flatten, return bool tensor in (n_grid**3, ), else in (n_grid, n_grid, n_grid)
        """
        if not self.contains_bitfield:
            return None

        if flatten:
            return self.bitfield.view(-1)
        else:
            return self.bitfield

    def reset_voxel_bitfield(self, occ=True):
        """Reset all the occupancy value

        Args:
            occ: single bool value for reset. By default occ.
        """
        dtype = self.bitfield.dtype
        device = self.bitfield.device
        if occ:
            self.bitfield = torch.ones_like(self.bitfield, dtype=dtype, device=device)
        else:
            self.bitfield = torch.zeros_like(self.bitfield, dtype=dtype, device=device)

    def update_bitfield(self, occupancy, ops='and'):
        """Update each voxel indicator by occupancy. new occupancy is the union of old and new.

        Args:
            occupancy: torch.bool in shape (n_grid**3, ) or (n_grid, n_grid, n_grid)
            ops: and/or/overwrite ops for update.
                     `And` only allows shrinking the volume.
                     `Or` can expand.
                     `Overwrite` directly use new one.
                      By default use and.

        Returns:
            bitfield: update by occupancy
        """
        if len(occupancy.shape) == 1:
            assert occupancy.shape[0] == self.get_n_voxel(), 'Occupancy should cover each voxel...'
            n_grid_occ = round(occupancy.shape[0]**(1 / 3))
            occupancy = occupancy.view(n_grid_occ, n_grid_occ, n_grid_occ)

        assert occupancy.dtype == torch.bool, 'Must input bool tensor'

        if ops == 'and':
            self.bitfield = torch.logical_and(self.bitfield, occupancy)  # only shrink
        elif ops == 'or':
            self.bitfield = torch.logical_or(self.bitfield, occupancy)  # can go larger
        elif ops == 'overwrite':
            self.bitfield = occupancy
        else:
            raise NotImplementedError('Ops {} not support for update bitfield...'.format(ops))

    def update_bitfield_by_voxel_idx(self, voxel_idx: torch.Tensor, occ=True):
        """Update each bit by voxel_idx.

        Args:
            voxel_idx: (B, 3) tensor of xyz index, should be in (0, n_grid)
            occ: The occupancy value to set. By default set to True.

        Returns:
            bitfield: update by voxel_idx
        """
        assert 0 <= torch.all(voxel_idx) < self.n_grid, 'Invalid voxel range is not allowed'
        uni_voxel_idx = self.get_unique_voxel_idx(voxel_idx)  # update unique voxel only
        self.bitfield[uni_voxel_idx[:, 0], uni_voxel_idx[:, 1], uni_voxel_idx[:, 2]] = occ  # update

    def get_n_occupied_voxel(self):
        """Get the num of occupied voxels"""
        return self.bitfield.sum()

    def get_occupied_voxel_idx(self, flatten=False):
        """Return the occupied voxel mask

        Args:
            flatten: whether to flatten the voxel idx

        Returns:
            occ_voxel_idx: if flatten, return bool tensor in (N_occ, ), else in (N_occ, 3)
        """
        occ_voxel_idx = torch.where(self.get_voxel_bitfield(flatten=True))[0]  # (N_occ,)
        if not flatten:
            occ_voxel_idx = self.convert_flatten_index_to_xyz_index(occ_voxel_idx, self.n_grid)  # (N_occ, 3)

        return occ_voxel_idx

    def get_occupied_grid_pts(self):
        """Return the occupied voxel corners

        Returns:
            occ_voxel_grid_pts: (N_occ, 8, 3) tensor, N occupied voxel's grid_pts
        """
        occ_voxel_idx = self.get_occupied_voxel_idx(flatten=False)  # (N_occ, 3)
        occ_grid_pts = self.get_grid_pts_by_voxel_idx(occ_voxel_idx)

        return occ_grid_pts

    def get_occupied_voxel_pts(self):
        """Return the occupied voxel center pts

        Returns:
            occ_voxel_pts: (N_occ, 3) tensor, N occupied voxel's center pts
        """
        occ_voxel_idx = self.get_occupied_voxel_idx(flatten=False)  # (N_occ, 3)
        occ_voxel_pts = self.get_voxel_pts_by_voxel_idx(occ_voxel_idx)

        return occ_voxel_pts

    def get_occupied_bounding_range(self):
        """Return the bounding volume range of the remaining voxels

        Returns:
            occ_aabb: (3, 2) bounding xyz range
        """
        if not torch.any(self.bitfield):  # empty field, return original volume
            return self.get_range()

        half_voxel_size = self.get_voxel_size(to_list=False) / 2.0  # (3,)
        offset = torch.cat([-half_voxel_size[:, None], half_voxel_size[:, None]], dim=1)  # (3, 2)

        # x dir
        x_flatten = self.bitfield.reshape(-1)
        x_voxel_idx = torch.where(x_flatten)[0]
        x_idx = x_voxel_idx[[0, -1]]  # (2,)
        x_idx = self.convert_flatten_index_to_xyz_index(x_idx, self.n_grid)
        x_voxel_pts = self.get_voxel_pts_by_voxel_idx(x_idx)[:, 0:1].permute(1, 0)  # (1, 2)
        x_range = x_voxel_pts + offset[0:1, :]  # (1, 2)

        # y dir
        y_flatten = self.bitfield.permute(1, 0, 2).reshape(-1)  # to yxz order
        y_voxel_idx = torch.where(y_flatten)[0]
        y_idx = y_voxel_idx[[0, -1]]  # (2,)
        y_idx = self.convert_flatten_index_to_xyz_index(y_idx, self.n_grid)
        y_idx = y_idx[:, [1, 0, 2]]  # to xyz order
        y_voxel_pts = self.get_voxel_pts_by_voxel_idx(y_idx)[:, 1:2].permute(1, 0)  # (1, 2)
        y_range = y_voxel_pts + offset[1:2, :]  # (1, 2)

        # z dir
        z_flatten = self.bitfield.permute(2, 0, 1).reshape(-1)  # to zxy order
        z_voxel_idx = torch.where(z_flatten)[0]
        z_idx = z_voxel_idx[[0, -1]]  # (2,)
        z_idx = self.convert_flatten_index_to_xyz_index(z_idx, self.n_grid)
        z_idx = z_idx[:, [1, 2, 0]]  # to xyz order
        z_voxel_pts = self.get_voxel_pts_by_voxel_idx(z_idx)[:, 2:3].permute(1, 0)  # (1, 2)
        z_range = z_voxel_pts + offset[2:3, :]  # (1, 2)

        occ_aabb = torch.cat([x_range, y_range, z_range], dim=0)  # (3, 2)

        return occ_aabb

    def get_occupied_bounding_corner(self):
        """Return the bounding volume range of the remaining voxels

        Returns:
            occ_corner: (8, 2) bounding xyz position like corner
        """
        occ_range = self.get_occupied_bounding_range()  # (3, 2)
        occ_corner = self.get_eight_permutation(occ_range[0][:, None], occ_range[1][:, None], occ_range[2][:, None])

        return occ_corner

    def check_pts_in_occ_voxel(self, pts: torch.Tensor):
        """Check whether pts in occupied voxels. Don't use it when n_pts * n_occ could be large.

        Args:
            pts: (B, 3), points to be check

        Returns:
            pts_in_occ_voxel: (B, ) whether each pts is in occupied voxels
        """
        voxel_idx, pts_in_occ_voxel = self.get_voxel_idx_from_xyz(pts)  # (B, 3), (B, )
        occ_voxel_idx = self.get_occupied_voxel_idx()  # (N_occ, 3)

        # flatten and find uniq elements
        valid_voxel_idx = voxel_idx[pts_in_occ_voxel]  # (B_valid, 3)

        # could take large memory
        check_voxel_in_occ_voxel = torch.logical_and(
            valid_voxel_idx.unsqueeze(1),
            occ_voxel_idx.unsqueeze(0),
        )  # (B_valid, N_occ, 3)
        check_voxel_in_occ_voxel = torch.all(check_voxel_in_occ_voxel, dim=-1)  # (B_valid, N_occ)
        check_voxel_in_occ_voxel = torch.any(check_voxel_in_occ_voxel, dim=-1)  # (B_valid, )
        pts_in_occ_voxel[pts_in_occ_voxel.clone()] = check_voxel_in_occ_voxel

        return pts_in_occ_voxel

    def set_up_voxel_opafield(self):
        """Set up the opacity as tensor in shape (n_grid, n_grid, n_grid), each value is the voxel density
        opacity is init as 0.0 at beginning(Nothing in side the voxel)

        Returns:
            opafield: a (n_grid, n_grid, n_grid) float tensor
        """
        assert self.n_grid is not None, 'Voxel bitfield must be set for known resolution volume'

        opafield = torch.zeros((self.n_grid, self.n_grid, self.n_grid), dtype=self.dtype)
        self.register_buffer('opafield', opafield)

    def update_opafield_by_voxel_idx(self, voxel_idx, opacity, ema=None):
        """Update the opafield by voxel_idx. Only update if original value >= 0

        Args:
            voxel_idx: (B, 3), voxel_idx that need to update. Assume no overlap
            opacity: (B, ), new opacity value in those voxels
            ema: If not none, update the value by moving average. Otherwise directly copy new value.
        """
        if ema is None:
            update_opa = opacity
        else:
            update_opa = torch.max(self.opafield[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]] * ema, opacity)

        # update only the voxels with opacity > 0
        update_opa = torch.where(
            self.opafield[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]] >= 0,
            update_opa,
            self.opafield[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]],
        )

        self.opafield[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]] = update_opa

    def get_mean_voxel_opacity(self):
        """Get the min opacity value of density_field"""
        return float(self.opafield.clamp(min=0).mean())

    def update_bitfield_by_opafield(self, threshold=0.01, ops='and'):
        """Update the bitfield(occupancy) by opafield that is large enough"""
        thres = min(self.get_mean_voxel_opacity(), threshold)
        update_opafield = (self.opafield >= thres)  # (n_grid, n_grid, n_grid)
        self.update_bitfield(update_opafield, ops)  # only valid occupancy is updated

    @staticmethod
    def get_lines_from_vertices(verts: torch.Tensor, n):
        """
        Args:
            verts: (n^3, 3) xyz points tensor
            n: num of point on each axis

        Returns:
            lines: 3*(n^3) lines of (2, 3) np array
        """
        assert verts.shape == (n**3, 3), 'Invalid input dim, should be (n^3, 3)'

        corner = verts.view(n, n, n, 3)
        lines = []
        for i in range(n):
            xz = torch_to_np(torch.cat([corner[i, 0, :, :][None, :], corner[i, n - 1, :, :][None, :]]))
            for k in range(xz.shape[1]):
                lines.append(xz[:, k, :])
            xy = torch_to_np(torch.cat([corner[i, :, 0, :][None, :], corner[i, :, n - 1, :][None, :]]))
            for k in range(xy.shape[1]):
                lines.append(xy[:, k, :])
            yz = torch_to_np(torch.cat([corner[0, i, :, :][None, :], corner[n - 1, i, :, :][None, :]]))
            for k in range(yz.shape[1]):
                lines.append(yz[:, k, :])

        return lines

    def get_bound_lines(self):
        """Get the outside bounding lines. for visual purpose.

        Returns:
            lines: list of 12(3*2^3), each is np array of (2, 3)
        """
        assert self.corner is not None, 'Please set the params first'
        lines = self.get_lines_from_vertices(self.corner, 2)

        return lines

    def get_dense_lines(self):
        """Get the bounding + inner lines. for visual purpose.

        Returns:
            lines: list of 3*(n+1)^3, each is np array of (2, 3)
        """
        assert self.grid_pts is not None, 'Please set the params first'
        lines = self.get_lines_from_vertices(self.grid_pts, self.n_grid + 1)

        return lines

    def get_occupied_lines(self):
        """Get the inner lines for occupied cells only

        Returns:
            lines: list of N_occ*12, each is np array of (2, 3)
        """
        assert self.bitfield is not None, 'Dont call it if you do not set bitfield'
        occ_voxel_grid_pts = self.get_occupied_grid_pts()  # (N_occ, 8, 3)
        lines = []
        for i in range(occ_voxel_grid_pts.shape[0]):
            voxel_line = self.get_lines_from_vertices(occ_voxel_grid_pts[i], 2)
            lines.extend(voxel_line)

        return lines

    @staticmethod
    def get_faces_from_vertices(verts: torch.Tensor, n):
        """
        Args:
            verts: (n^3, 3) xyz points tensor
            n: num of point on each axis

        Returns:
            faces: (n(n-1)^2*3, 4, 3) np array
        """
        assert verts.shape == (n**3, 3), 'Invalid input dim, should be (n^3, 3)'

        corner = verts.view(n, n, n, 3)
        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                xy = torch.cat([
                    corner[i, j, :, :][None, :], corner[i, j + 1, :, :][None, :], corner[i + 1, j, :, :][None, :],
                    corner[i + 1, j + 1, :, :][None, :]
                ])
                for k in range(xy.shape[1]):
                    faces.append(xy[:, k, :][None, :])
                xz = torch.cat([
                    corner[i, :, j, :][None, :], corner[i, :, j + 1, :][None, :], corner[i + 1, :, j, :][None, :],
                    corner[i + 1, :, j + 1, :][None, :]
                ])
                for k in range(xz.shape[1]):
                    faces.append(xz[:, k, :][None, :])
                yz = torch.cat([
                    corner[:, i, j, :][None, :], corner[:, i, j + 1, :][None, :], corner[:, i + 1, j, :][None, :],
                    corner[:, i + 1, j + 1, :][None, :]
                ])
                for k in range(yz.shape[1]):
                    faces.append(yz[:, k, :][None, :])

        faces = torch_to_np(torch.cat(faces))

        return faces

    def get_bound_faces(self):
        """Get the outside bounding surface. for visual purpose.

        Returns:
            faces: (6, 4, 3) np array
        """
        assert self.corner is not None, 'Please set the params first'
        faces = self.get_faces_from_vertices(self.corner, 2)

        return faces

    def get_dense_faces(self):
        """Get the bounding + inner faces. for visual purpose.

        Returns:
            faces: ((n_grid^2(n_grid+1)*3, 4, 3) np array
        """
        assert self.grid_pts is not None, 'Please set the params first'
        faces = self.get_faces_from_vertices(self.grid_pts, self.n_grid + 1)

        return faces

    def get_occupied_faces(self):
        """Get the inner faces for occupied cells only

        Returns:
            faces: (N_occ*6, 4, 3) np array
        """
        assert self.bitfield is not None, 'Dont call it if you do not set bitfield'
        occ_voxel_grid_pts = self.get_occupied_grid_pts()  # (N_occ, 8, 3)
        faces = []
        for i in range(occ_voxel_grid_pts.shape[0]):
            voxel_face = self.get_faces_from_vertices(occ_voxel_grid_pts[i], 2)
            faces.extend(voxel_face)
        faces = np.concatenate(faces, axis=0).reshape(-1, 4, 3)

        return faces
