# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from . import ENCODER_REGISTRY
from arcnerf.geometry.volume import Volume


@ENCODER_REGISTRY.register()
class DenseGridEmbedder(nn.Module):
    """Use a dense grid to embed all the xyz. You can extract sigma directly from dense volume or feat as well
    This can be only used for xyz positions, but not direction
    """

    def __init__(
        self,
        input_dim=3,
        n_grid=128,
        origin=(0, 0, 0),
        side=1.5,
        xlen=None,
        ylen=None,
        zlen=None,
        dtype=torch.float32,
        radius_init=None,
        include_input=False,
        W_feat=0,
        *args,
        **kwargs
    ):
        """
        Args:
            The following are for volume:
                n_grid: N of volume/line seg on each side. Each side is divided into n_grid seg with n_grid+1 pts.
                    total num of volume is n_grid**3, total num of grid_pts is (n_grid+1)**3.
                origin: origin point(centroid of cube), a tuple of 3
                side: each side len, if None, use xyzlen. If exist, use side only. By default 1.5.
                xlen: len of x dim, if None use side
                ylen: len of y dim, if None use side
                zlen: len of z dim, if None use side
                dtype: dtype of params. By default is torch.float32
            radius_init: If not None, init the geo value as sdf of a sphere
            include_input: if True, raw input is included in the embedding. Appear at beginning. By default is True
            W_feat: output feature if W_feat > 0

        Returns:
            Embedded inputs with shape:
                1 + include_input * input_dim + W_feat
        """
        super(DenseGridEmbedder, self).__init__()

        assert input_dim == 3, 'HashGridEmbedder should has input_dim==3...'
        assert W_feat >= 0, 'Should not input a negative W_feat'
        self.input_dim = input_dim
        self.include_input = include_input

        # set volume with base res
        self.volume = Volume(n_grid=n_grid, origin=origin, side=side, xlen=xlen, ylen=ylen, zlen=zlen, dtype=dtype)
        self.n_grid = n_grid

        # set up dense grid params
        self.grid_value_param, self.grid_feat_param = self.init_volume_params(W_feat, radius_init)
        self.radius = radius_init
        self.W_feat = W_feat

        self.out_dim = 1 + include_input * input_dim + W_feat

    def get_output_dim(self):
        """Get output dim"""
        return self.out_dim

    def init_volume_params(self, W_feat, radius_init, std=1e-4):
        """Init volume params for density and feature"""
        grid_value_param = nn.Parameter(torch.zeros(((self.n_grid + 1)**3, 1)))
        if radius_init is not None:
            grid_pts = self.volume.get_grid_pts()
            sdf = torch.norm(grid_pts, dim=-1, keepdim=True) - radius_init  # (n_grid^3, 1)
            grid_value_param.data = sdf
        else:
            nn.init.normal_(grid_value_param)

        grid_feat_param = None
        if W_feat > 0:
            grid_feat_param = nn.Parameter(torch.zeros(((self.n_grid + 1)**3, W_feat)))
            nn.init.uniform_(grid_feat_param, -std, std)

        return grid_value_param, grid_feat_param

    def forward(self, xyz: torch.Tensor):
        """
        Args:
            xyz: tensor of shape (B, 3), xyz position

        Returns:
            out: tensor of shape (B, out_dim=1 + include_inputs * input_dim + W_feat)
        """
        assert len(xyz.shape) == 2 and xyz.shape[-1] == 3, 'Must be (B, 3) tensor'

        out = []

        # set up geo_value output
        if self.radius is None:
            geo_value = (xyz.clone() * 0.0)[:, :1]  # (B, 1)
        else:
            geo_value = (xyz.clone() * 0.0)[:, :1] + 1.0  # (B, 1)
            out *= (self.radius_init * 2.0)

        # set up feat output
        feat = None
        if self.W_feat > 0:
            feat = (xyz.clone() * 0.0)[:, :1].repeat(1, self.W_feat)  # (B, W_feat)

        # get voxel grid info
        voxel_grid = self.volume.get_voxel_grid_info_from_xyz(xyz)
        voxel_idx, valid_idx, grid_pts_weights_valid = voxel_grid[0], voxel_grid[1], voxel_grid[-1]

        if grid_pts_weights_valid is not None:  # any points in volume
            geo_value[valid_idx] = self.volume.interpolate(
                self.grid_value_param, grid_pts_weights_valid, voxel_idx[valid_idx]
            )  # (B_valid, W_vol)

            if self.W_feat > 0:
                feat[valid_idx] = self.volume.interpolate(
                    self.grid_feat_param, grid_pts_weights_valid, voxel_idx[valid_idx]
                )  # (B_valid, W_vol)

        # append the result
        out.append(geo_value)  # (B, 1)
        if self.include_input:
            out.append(xyz)  # (B, 3)
        if feat is not None:
            out.append(feat)  # (B, W_feat)

        out = torch.cat(out, dim=-1)  # (B, 1 + include_input * input_dim + W_feat)

        return out
