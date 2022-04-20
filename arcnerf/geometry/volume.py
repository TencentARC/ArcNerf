# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from common.utils.torch_utils import torch_to_np


class Volume(nn.Module):
    """A volume with custom operation"""

    def __init__(self, n_grid, origin=(0, 0, 0), side=None, xlen=None, ylen=None, zlen=None, requires_grad=False):
        """
        Args:
            n_grid: N of volume/line seg on each side. Each side is divided into n_grid seg with n_grid+1 pts.
                    total num of volume is n_grid**3, total num of grid_pts is (n_grid+1)**3.
            origin: origin point(centroid of cube), a tuple of 3
            side: each side len, if None, use xyzlen. If exist, use side only
            xlen: len of x dim, if None use side
            ylen: len of y dim, if None use side
            zlen: len of z dim, if None use side
            requires_grad: whether the parameters requires grad
        """
        super(Volume, self).__init__()
        self.requires_grad = requires_grad

        # set nn params
        self.n_grid = n_grid
        self.origin = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]), requires_grad=self.requires_grad)
        self.xlen = nn.Parameter(torch.tensor([0.0]), requires_grad=self.requires_grad)
        self.ylen = nn.Parameter(torch.tensor([0.0]), requires_grad=self.requires_grad)
        self.zlen = nn.Parameter(torch.tensor([0.0]), requires_grad=self.requires_grad)
        self.range = None
        self.corner = None
        self.grid_pts = None
        self.volume_pts = None

        # set real value
        if origin is not None and (side is not None or all([length is not None for length in [xlen, ylen, zlen]])):
            self.set_params(origin, side, xlen, ylen, zlen)

    def set_params(self, origin, side, xlen, ylen, zlen):
        """you can call outside to reset the params"""
        assert side is not None or all([length is not None for length in [xlen, ylen, zlen]]),\
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

    @torch.no_grad()
    def set_origin(self, origin=(0.0, 0.0, 0.0)):
        """Set the origin """
        self.origin[0] = origin[0]
        self.origin[1] = origin[1]
        self.origin[2] = origin[2]

    def get_origin(self):
        """
        Returns:
            origin: tensor(3, )
        """
        return self.origin

    def cal_range(self):
        """Cal the xyz range(min, max) from origin and sides. range is (3, 2) tensor"""
        xyz_min = self.origin - torch.cat([self.xlen, self.ylen, self.zlen]) / 2.0
        xyz_max = self.origin + torch.cat([self.xlen, self.ylen, self.zlen]) / 2.0
        self.range = torch.cat([xyz_min[:, None], xyz_max[:, None]], dim=-1)

    def get_range(self):
        """Get the xyz range in (3, 2)"""
        return self.range

    def cal_corner(self):
        """Cal the eight corner pts given origin and sides. corner is (8, 3) tensor """
        x, y, z = self.range[0][:, None], self.range[1][:, None], self.range[2][:, None]  # (2, 1)
        x_r = x.unsqueeze(0).repeat(2, 1, 1)  # (2, 2, 1)
        y_r = y.unsqueeze(1).repeat(1, 2, 1)  # (2, 2, 1)
        xy_r = torch.cat([x_r, y_r], -1).view(-1, 2)  # (4, 2)
        xy_r = xy_r.unsqueeze(0).repeat(2, 1, 1)  # (2, 4, 2)
        z_r = z.unsqueeze(1).repeat(1, 4, 1)  # (2, 4, 1)
        self.corner = torch.cat([xy_r, z_r], -1).view(-1, 3)  # (8, 3)

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
        self.grid_pts = grid_pts.view(-1, 3)  # (n+1^3, 3)

    def get_grid_pts(self, in_grid=False):
        """Get ((self.n_grid+1)^3, 3) grid pts. If in_grid, return (n_grid+1, n_grid+1, n_grid+1, 3) """
        if in_grid:
            return self.grid_pts.view(self.n_grid + 1, self.n_grid + 1, self.n_grid + 1, 3)
        else:
            return self.grid_pts

    def cal_volume_pts(self):
        """Cal all the volume center pts. tensor ((self.n_grid)^3, 3) """
        v_x, v_y, v_z = self.get_volume_size()  # volume size
        x = torch.linspace(float(self.range[0, 0]) + 0.5 * v_x, float(self.range[0, 1]) - 0.5 * v_x, self.n_grid)  # (n)
        y = torch.linspace(float(self.range[1, 0]) + 0.5 * v_y, float(self.range[1, 1]) - 0.5 * v_y, self.n_grid)  # (n)
        z = torch.linspace(float(self.range[2, 0]) + 0.5 * v_z, float(self.range[2, 1]) - 0.5 * v_z, self.n_grid)  # (n)

        volume_pts = torch.stack(torch.meshgrid(x, y, z), -1)  # (n, n, n, 3)
        self.volume_pts = volume_pts.view(-1, 3)  # (n^3, 3)

    def get_volume_pts(self, in_grid=False):
        """Get ((self.n_grid)^3, 3) volume pts. If in_grid, return (n_grid, n_grid, n_grid, 3) """
        if in_grid:
            return self.volume_pts.view(self.n_grid, self.n_grid, self.n_grid, 3)
        else:
            return self.volume_pts

    def get_volume_size(self):
        """Get volume size on each side. Good for marching cube"""
        x_s = float(self.xlen) / float(self.n_grid)
        y_s = float(self.ylen) / float(self.n_grid)
        z_s = float(self.zlen) / float(self.n_grid)

        return x_s, y_s, z_s

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
