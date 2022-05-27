# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from common.utils.torch_utils import torch_to_np


class Volume(nn.Module):
    """A volume with custom operation"""

    def __init__(
        self,
        n_grid,
        origin=(0, 0, 0),
        side=None,
        xlen=None,
        ylen=None,
        zlen=None,
        dtype=torch.float32,
        requires_grad=False
    ):
        """
        Args:
            n_grid: N of volume/line seg on each side. Each side is divided into n_grid seg with n_grid+1 pts.
                    total num of volume is n_grid**3, total num of grid_pts is (n_grid+1)**3.
            origin: origin point(centroid of cube), a tuple of 3
            side: each side len, if None, use xyzlen. If exist, use side only
            xlen: len of x dim, if None use side
            ylen: len of y dim, if None use side
            zlen: len of z dim, if None use side
            dtype: dtype of params. By default is torch.float32
            requires_grad: whether the parameters requires grad. If True, waste memory for graphic.
        """
        super(Volume, self).__init__()
        self.requires_grad = requires_grad

        # set nn params
        self.n_grid = n_grid
        self.origin = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], dtype=dtype), requires_grad=self.requires_grad)
        self.xlen = nn.Parameter(torch.tensor([0.0], dtype=dtype), requires_grad=self.requires_grad)
        self.ylen = nn.Parameter(torch.tensor([0.0], dtype=dtype), requires_grad=self.requires_grad)
        self.zlen = nn.Parameter(torch.tensor([0.0], dtype=dtype), requires_grad=self.requires_grad)
        self.range = None
        self.corner = None
        self.grid_pts = None
        self.volume_pts = None

        # set real value
        if origin is not None and (side is not None or all([length is not None for length in [xlen, ylen, zlen]])):
            self.set_params(origin, side, xlen, ylen, zlen)

    def set_params(self, origin, side, xlen, ylen, zlen):
        """you can call outside to reset the params"""
        assert side is not None or all([length is not None for length in [xlen, ylen, zlen]]), \
            'Specify at least side or xyzlen'
        self.set_origin(origin)
        self.set_len(side, xlen, ylen, zlen)
        self.cal_range()
        self.cal_corner()
        self.cal_grid_pts()
        self.cal_volume_pts()

    @torch.no_grad()
    def set_len(self, side, xlen, ylen, zlen):
        """Set len of each dim"""
        if side is not None:
            self.xlen[0] = side
            self.ylen[0] = side
            self.zlen[0] = side
        else:
            self.xlen[0] = xlen
            self.ylen[0] = ylen
            self.zlen[0] = zlen

    def get_len(self):
        """Return len of each dim, in tuple of float num"""
        return float(self.xlen[0]), float(self.ylen[0]), float(self.zlen[0]),

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
        xyz_min = self.origin - torch.cat([self.xlen, self.ylen, self.zlen]) / 2.0
        xyz_max = self.origin + torch.cat([self.xlen, self.ylen, self.zlen]) / 2.0
        v_range = torch.cat([xyz_min[:, None], xyz_max[:, None]], dim=-1)
        self.range = nn.Parameter(v_range.clone().detach(), requires_grad=self.requires_grad)

    def get_range(self):
        """Get the xyz range in (3, 2)"""
        return self.range

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
        corner = self.get_eight_permutation(x, y, z)
        self.corner = nn.Parameter(corner.clone().detach(), requires_grad=self.requires_grad)

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
        grid_pts = grid_pts.view(-1, 3)  # (n+1^3, 3)
        self.grid_pts = nn.Parameter(grid_pts.clone().detach(), requires_grad=self.requires_grad)

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
        volume_pts = volume_pts.view(-1, 3)  # (n^3, 3)
        self.volume_pts = nn.Parameter(volume_pts.clone().detach(), requires_grad=self.requires_grad)

    def get_volume_pts(self, in_grid=False):
        """Get ((self.n_grid)^3, 3) volume pts. If in_grid, return (n_grid, n_grid, n_grid, 3) """
        if in_grid:
            return self.volume_pts.view(self.n_grid, self.n_grid, self.n_grid, 3)
        else:
            return self.volume_pts

    def get_voxel_size(self):
        """Get each voxel size on each side. Good for marching cube"""
        xyz_s = self.get_grid_pts(True)[1, 1, 1, :] - self.get_grid_pts(True)[0, 0, 0, :]  # (3,)
        x_s, y_s, z_s = float(xyz_s[0]), float(xyz_s[1]), float(xyz_s[2])

        return x_s, y_s, z_s

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
    def check_pts_in_grid_boundary(pts, grid_pts):
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
        pts_min_req = torch.BoolTensor(pts >= grid_pts_expand.min(dim=1)[0]).to(pts.device)
        pts_max_req = torch.BoolTensor(pts < grid_pts_expand.max(dim=1)[0]).to(pts.device)
        pts_in_boundary = torch.logical_and(pts_min_req, pts_max_req)  # (B, 3)
        pts_in_boundary = torch.all(pts_in_boundary, dim=-1)  # (B, )

        return pts_in_boundary

    def get_voxel_idx_from_xyz(self, pts):
        """Get the voxel idx from xyz pts. Directly calculate the offset index in each dim.

        Args:
            pts: xyz position of points. (B, 3) tensor of xyz

        Returns:
            voxel_idx: return the voxel index in (B, 3) of each xyz index, range in (0, n_grid).
                       if not in any voxel, return (-1, -1, -1) for this pts. torch.long
            valid_idx: (B, ) mask that pts are in the volume. torch.BoolTensor
        """
        dtype = pts.dtype
        device = pts.device

        x_s, y_s, z_s = self.get_voxel_size()
        voxel_size = torch.Tensor([x_s, y_s, z_s]).type(dtype).to(device).unsqueeze(0)  # (1, 3)
        start_point = self.get_range()[:, 0].unsqueeze(0)  # min xyz (1, 3)
        voxel_idx = (pts - start_point) / voxel_size  # (B, 3)

        valid_idx = torch.logical_and(
            torch.all(voxel_idx >= 0, dim=1), torch.all(voxel_idx < float(self.n_grid), dim=1)
        )

        voxel_idx[~valid_idx] = -1
        voxel_idx = torch.floor(voxel_idx).type(torch.long)

        return voxel_idx, valid_idx

    def get_grid_pts_idx_by_voxel_idx(self, voxel_idx):
        """Get the grid pts index from voxel idx

        Args:
            voxel_idx: (B, 3) tensor of xyz index, should be in (0, n_grid)

        Returns:
            grid_pts_idx_by_idx: select grid_pts index by voxel_idx, (B, 8, 3), each index of grid
        """
        assert 0 <= voxel_idx.min() <= voxel_idx.max() < self.n_grid, 'Voxel idx out of boundary'

        permute_index = self.get_eight_permutation_index().to(voxel_idx.device)  # (8, 3)

        # add 0,1 offset to voxel_idx as grid_pts index
        grid_pts_idx = voxel_idx.unsqueeze(1) + permute_index.unsqueeze(0)  # (B, 8, 3)
        grid_pts_idx = grid_pts_idx.view(-1, 3)  # (B*8, 3)
        grid_pts_idx = self.convert_xyz_index_to_flatten_index(grid_pts_idx, self.n_grid + 1)  # (B*8, )

        return grid_pts_idx.view(-1, 8)

    def get_grid_pts_by_voxel_idx(self, voxel_idx):
        """Get the grid pts xyz from voxel idx

        Args:
            voxel_idx: (B, 3) tensor of xyz index, should be in (0, n_grid)

        Returns:
            grid_pts_by_voxel_idx: select grid_pts xyz values by voxel_idx, (B, 8, 3), each pts of grid
        """
        grid_pts = self.get_grid_pts()  # (n+1^3, 3)
        grid_pts_idx = self.get_grid_pts_idx_by_voxel_idx(voxel_idx)  # (B, 8)

        # select by idx
        grid_pts_by_voxel_idx = self.collect_grid_pts_values(grid_pts, grid_pts_idx)  # (B, 8, 3)

        return grid_pts_by_voxel_idx

    def cal_weights_to_grid_pts(self, pts, grid_pts):
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
        weights = []
        for idx in range(permute_index.shape[0]):
            weights.append(
                (permute_index[idx][0] * (w_xyz[:, 0:1]) + (1 - permute_index[idx][0]) * (1 - w_xyz[:, 0:1])) *
                (permute_index[idx][1] * (w_xyz[:, 1:2]) + (1 - permute_index[idx][1]) * (1 - w_xyz[:, 1:2])) *
                (permute_index[idx][2] * (w_xyz[:, 2:3]) + (1 - permute_index[idx][2]) * (1 - w_xyz[:, 2:3]))
            )
        weights = torch.cat(weights, dim=-1)  # (B, 8)

        return weights

    def interpolate(self, values, weights, voxel_idx):
        """Interpolate values by getting value on grid_pts in each voxel and multiply weights

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
        weights_expand = weights.view(*weights.shape, *(1, ) * (len(values.shape) - 1))  # expand like values
        values_by_weights = values_on_grid_pts * weights_expand  # (B, 8, ...)
        values_by_weights = values_by_weights.sum(1)  # (B, ...)

        return values_by_weights

    @staticmethod
    def convert_xyz_index_to_flatten_index(xyz, n):
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
    def convert_flatten_index_to_xyz_index(flatten_index, n):
        """Convert flatten index to xyz index

        Args:
            flatten_index: (B, ) flatten index in torch.long
            n: num of offset on each dim. Generally use n_grid or n_grid + 1

        Returns:
            xyz: (B, 3) index in torch.long
        """
        xyz_index = torch.zeros((flatten_index.shape[0], 3), dtype=torch.int).to(flatten_index.device)

        for i in range(3):
            xyz_index[:, 2 - i] = flatten_index % n
            flatten_index = flatten_index // n

        return xyz_index

    def collect_grid_pts_values(self, values, grid_pts_idx):
        """Collect values on grid_pts(xyz, feature, value) by grid_pts_idx

        Args:
            values: (n_grid+1^3, ...) values, all the grid_pts_values. Only the first shape is required.
            grid_pts_idx: (B, 8), grid pts idx of each sample

        Returns:
            values_by_grid_pts_idx: (B, 8, ...), selected values on each grid_pts,
                                    shape[2:] same as values.shape[1:]
        """
        assert values.shape[0] == (self.n_grid + 1)**3, 'Invalid dim of values...'

        # select by idx
        values_by_grid_pts_idx = values[grid_pts_idx, ...]  # (B, 8, ...)

        return values_by_grid_pts_idx

    def get_bound_lines(self):
        """Get the outside bounding lines. for visual purpose.

        Returns:
            lines: list of 12(3*2^3), each is np array of (2, 3)
        """
        assert self.corner is not None, 'Please set the params first'
        lines = self.get_lines_from_vertices(self.corner, 2)

        return lines

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

    def get_dense_lines(self):
        """Get the bounding + inner lines. for visual purpose.

        Returns:
            lines: list of 3*(n+1)^3, each is np array of (2, 3)
        """
        assert self.grid_pts is not None, 'Please set the params first'
        lines = self.get_lines_from_vertices(self.grid_pts, self.n_grid + 1)

        return lines

    def get_bound_faces(self):
        """Get the outside bounding surface. for visual purpose.

        Returns:
            faces: (6, 4, 3) np array
        """
        assert self.corner is not None, 'Please set the params first'
        faces = self.get_faces_from_vertices(self.corner, 2)

        return faces

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

    def get_dense_faces(self):
        """Get the bounding + inner faces. for visual purpose.

        Returns:
            faces: ((n_grid^2(n_grid+1)*3, 4, 3) np array
        """
        assert self.grid_pts is not None, 'Please set the params first'
        faces = self.get_faces_from_vertices(self.grid_pts, self.n_grid + 1)

        return faces
