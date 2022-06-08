# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from . import MODULE_REGISTRY
from .activation import get_activation
from .base_netwok import BaseGeoNet, BaseRadianceNet
from .embed import Embedder
from .linear import DenseLayer
from arcnerf.geometry.volume import Volume


@MODULE_REGISTRY.register()
class VolGeoNet(BaseGeoNet):
    """Dense Volume Net with voxels, modeling geometry.
    Each grid pts (N+1)^3 store some representation. Each pts/dir is interpolated from the grid pts and get result.
    Shallow network is support after grid interpolation.

    Sample size for n_grid^3: 64^3 = 2e-4G / 128^3 = 2e-3G / 256^3 = 0.015G / 512^3 = 0.125G

    Several Mode is allowed for processing:
    - use_nn is False:
        - return (W_vol, W_feat_vol) as the result from voxel interpolation. Any one can be < 0
    - use_nn is True:
        - You have several inputs for network: feature from volume/xyz/xyz_embed, depends on choice
        - The output from network is (1, W_feat), which replace the output from voxel interpolation.

    Ref: NSVF/ Plenoctree/ Plenoxels, etc. But pruning like operation is not support in dense volume.
    """

    def __init__(
        self,
        W_vol=1,
        W_feat_vol=256,
        geometric_init=True,
        radius_init=0.5,
        n_grid=128,
        origin=(0, 0, 0),
        side=1.5,
        xlen=None,
        ylen=None,
        zlen=None,
        dtype=torch.float32,
        use_nn=False,
        W=256,
        D=2,
        include_input=True,
        input_ch=3,
        embed_freq=6,
        W_feat=256,
        act_cfg=None,
        weight_norm=False,
        *args,
        **kwargs
    ):
        """
        Args:
            W_feat_vol: Num of feature store at each grid pts. By default 256.
            geometric_init: If True, init sdf like a sphere. by default True.
            radius_init: radius of sphere used for geometric_init. Output sdf can be init like sphere. By fault 0.5.
                         Nerf do not use it. Only for surface modeling methods(idr/neus/volsdf/unisurf)
            The following are for volume:
                n_grid: N of volume/line seg on each side. Each side is divided into n_grid seg with n_grid+1 pts.
                    total num of volume is n_grid**3, total num of grid_pts is (n_grid+1)**3.
                origin: origin point(centroid of cube), a tuple of 3
                side: each side len, if None, use xyzlen. If exist, use side only. By default 1.5.
                xlen: len of x dim, if None use side
                ylen: len of y dim, if None use side
                zlen: len of z dim, if None use side
                dtype: dtype of params. By default is torch.float32
            use_nn: If True, adopt a shallow nn network at the end. It takes the feature from volume and
                    output final results. By default False
            W: mlp hidden layer size, by default 256
            D: num of hidden layers, by default 2
            include_input: include input with feat embed from voxels. By default True.
            input_ch: input channel num, by default 3(xyz). It is the dim before embed.
            embed_freq: embedding freq. by default 6. (Nerf use 10)
                        output dim will be input_ch * (freq * 2 + 1). 0 means not embed.
            W_feat: output feature from
            act_cfg: cfg obj for selecting activation. None for relu.
                     For surface modeling, usually use SoftPlus(beta=100)
            weight_norm: If weight_norm, do extra weight_normalization for linear layer. By default False.
                         Nerf do not use it. Only for surface modeling methods(idr/neus/volsdf/unisurf)
        """
        super(VolGeoNet, self).__init__()

        # volume params
        self.W_vol = W_vol
        self.W_feat_vol = W_feat_vol
        self.n_grid = n_grid

        # nn params
        self.use_nn = use_nn
        self.W = W
        self.D = D
        self.W_feat = W_feat
        self.weight_norm = weight_norm
        self.include_input = include_input

        self.radius_init = radius_init
        self.geometric_init = geometric_init
        self.dtype = dtype

        # set volume.
        self.volume = Volume(n_grid=n_grid, origin=origin, side=side, xlen=xlen, ylen=ylen, zlen=zlen, dtype=dtype)
        self.n_grid_pts = (n_grid + 1)**3
        self.n_voxel = n_grid**3

        # set grid_pts param for geo value and feature, in (n_grid+1)^3 dim, for volume
        self.grid_value_param = None
        if W_vol > 0:
            self.grid_value_param = nn.Parameter(torch.zeros(((n_grid + 1)**3, W_vol), dtype=dtype))
        self.grid_feature_param = None
        if W_feat_vol > 0:
            self.grid_feature_param = nn.Parameter(torch.zeros(((n_grid + 1)**3, W_feat_vol), dtype=dtype))

        # init vol param
        self.init_vol_param()

        # set shallow nn network
        self.layers = None
        if use_nn and D > 0 and (W_feat_vol > 0 or include_input):
            self.embed_fn, self.layers = self.setup_network(
                W, D, W_feat_vol, W_feat, act_cfg, include_input, input_ch, embed_freq, *args, **kwargs
            )

    def get_volume(self):
        """Get the dense volume"""
        return self.volume

    def setup_network(self, W, D, W_in, W_feat, act_cfg, include_input, input_ch, embed_freq, *args, **kwargs):
        """Set up the nn network"""
        layers = []

        # input layer
        embed_fn, embed_dim = None, 0
        if include_input:
            embed_fn = Embedder(input_ch, embed_freq, *args, **kwargs)
            embed_dim = embed_fn.get_output_dim()

        for i in range(D + 1):
            in_dim = W_in + embed_dim if i == 0 else W
            out_dim = W if i != D else 1 + max(W_feat, 0)

            if i != D:
                layer = DenseLayer(in_dim, out_dim, activation=get_activation(act_cfg))
            else:
                layer = nn.Linear(in_dim, out_dim)

            if self.geometric_init:
                if i == D:  # last layer
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    nn.init.constant_(layer.bias[:1], -self.radius_init)  # bias only for geo value
                elif include_input and embed_freq > 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    if i == 0:  # first layer, [feature, x, embed_x], do not init for embed_x
                        torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                        torch.nn.init.constant_(layer.weight[:, -(embed_dim - input_ch):], 0.0)
                    else:
                        nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                else:
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            # extra weight normalization
            if self.weight_norm:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        layers = nn.ModuleList(layers)

        return embed_fn, layers

    def init_vol_param(self):
        """Init param on the grid pts"""
        # init feature as normal
        if self.grid_feature_param is not None:
            nn.init.normal_(self.grid_feature_param)

        # init the value
        if self.grid_value_param is not None:
            if self.geometric_init:  # init the geo value such that it is sdf of sphere
                grid_pts = self.volume.get_grid_pts()  # (n_grid^3, 3)
                sdf = torch.norm(grid_pts, dim=-1, keepdim=True) - self.radius_init  # (n_grid^3, 1)
                self.grid_value_param.data = sdf
            else:
                nn.init.normal_(self.grid_value_param)

    def forward(self, x):
        """
        For any pts xyz, if it is in the voxel, use interpolation for the pts,
                         else if not in the volume, assign 0 to all result.
        Args:
            x: torch.tensor (B, input_ch), the first three must be xyz position of pts

        Returns:
            out: tensor in shape (B, 1) for geometric value(sdf, sigma, occ).
            out_feat: tensor in shape (B, W_feat) if W_feat > 0. None if W_feat <= 0
        """
        dtype = x.dtype
        device = x.device
        n_pts = x.shape[0]

        # get valid voxel idx from pts xyz
        voxel_idx, valid_idx = self.volume.get_voxel_idx_from_xyz(x)  # (B, 3), xyz index of volume
        assert voxel_idx.max() < self.n_grid, 'Voxel idx exceed boundary...'

        out, out_feat = None, None

        if torch.any(valid_idx):  # any points in volume
            # get valid grid pts position
            grid_pts_valid = self.volume.get_grid_pts_by_voxel_idx(voxel_idx[valid_idx])  # (B_valid, 8, 3)
            # calculate weights to 8 grid_pts by inverse distance
            grid_pts_weights_valid = self.volume.cal_weights_to_grid_pts(x[valid_idx], grid_pts_valid)  # (B_valid, 8)

        # select geo value by index
        if self.W_vol > 0:
            if self.geometric_init:  # sdf outside is +
                out = torch.ones((n_pts, self.W_vol), dtype=dtype).to(device)  # (B, W_vol) in generally should be 1
                out *= (self.radius_init * 2)
            else:
                out = torch.zeros((n_pts, self.W_vol), dtype=dtype).to(device)  # (B, W_vol) in generally should be 1
            if torch.any(valid_idx):
                out[valid_idx] = self.volume.interpolate(
                    self.grid_value_param, grid_pts_weights_valid, voxel_idx[valid_idx]
                )  # (B_valid, W_vol)

        # select feature by index, this could be slow for large tensor multiplication even in gpu
        if self.W_feat_vol > 0:
            # large feat transfer from cpu to gpu takes time. Use clone save time.
            out_feat = (x.clone() * 0.0)[:, :1].repeat(1, self.W_feat_vol)  # (B, W_feat_vol)
            if torch.any(valid_idx):  # any points in volume
                out_feat[valid_idx] = self.volume.interpolate(
                    self.grid_feature_param, grid_pts_weights_valid, voxel_idx[valid_idx]
                )  # (B_valid, W_feat_vol)

        # nn forward, you could use W_feat_vol/xyz/xyz_embed as input, any of them can be missed
        if self.layers is not None:
            # get input
            out = []
            if self.W_feat_vol > 0:
                out.append(out_feat)
            if self.include_input:
                out.append(self.embed_fn(x))

            out = torch.cat(out, dim=-1)  # (B, W_feat_vol + embed_dim)
            for i in range(self.D + 1):
                out = self.layers[i](out)

            if self.W_feat <= 0:
                out_feat = None
            else:
                out, out_feat = out[:, :1], out[:, 1:]

        return out, out_feat


@MODULE_REGISTRY.register()
class VolRadianceNet(BaseRadianceNet):
    """Dense Volume Net with voxels, modeling radiance.
    Each grid pts (N+1)^3 store some representation. Each pts/dir is interpolated from the grid pts and get result.
    Shallow network is support after grid interpolation.

    Sample size for n_grid^3: 64^3 = 2e-4G / 128^3 = 2e-3G / 256^3 = 0.015G / 512^3 = 0.125G

    Several Mode is allowed for processing:
    - use_nn is False:
        - return (3, ) as the rgb result from voxel interpolation. Should be positive
    - use_nn is True:
        - You have several inputs for network: feature from volume/xyz/xyz_embed/view/view_embed/normal,
        depends on choice

    Ref: NSVF/ Plenoctree/ Plenoxels, etc. But pruning like operation is not support in dense volume.
    """

    def __init__(
        self,
        mode='vf',
        n_grid=128,
        origin=(0, 0, 0),
        side=1.5,
        xlen=None,
        ylen=None,
        zlen=None,
        dtype=torch.float32,
        use_nn=False,
        W=256,
        D=4,
        input_ch_pts=3,
        input_ch_view=3,
        embed_freq_pts=6,
        embed_freq_view=4,
        W_feat_in=256,
        act_cfg=None,
        weight_norm=False,
        *args,
        **kwargs
    ):
        """
        Args:
            mode: 'p':points(xyz) - 'v':view_dirs - 'n':normals - 'f'-geo_feat
                  Should be a str combining all used inputs. By default 'vf', use view_dir and geo_feat like nerf.
            The following are for volume:
                n_grid: N of volume/line seg on each side. Each side is divided into n_grid seg with n_grid+1 pts.
                    total num of volume is n_grid**3, total num of grid_pts is (n_grid+1)**3.
                origin: origin point(centroid of cube), a tuple of 3
                side: each side len, if None, use xyzlen. If exist, use side only. By default 1.5.
                xlen: len of x dim, if None use side
                ylen: len of y dim, if None use side
                zlen: len of z dim, if None use side
                dtype: dtype of params. By default is torch.float32
            use_nn: If True, adopt a shallow nn network at the end. It takes the feature from volume and
                    output final results. By default False
            W: mlp hidden layer size, by default 256
            D: num of hidden layers, by default 4
            input_ch_pts: input channel num for points, by default 3(xyz). It is the dim before embed.
            input_ch_view: input channel num for view_dirs, by default 3. It is the dim before embed.
            embed_freq_pts: embedding freq for pts. by default 6.
                            output dim will be input_ch * (freq * 2 + 1). 0 means not embed.
            embed_freq_view: embedding freq for view_dir. by default 4.
                            output dim will be input_ch * (freq * 2 + 1). 0 means not embed.
            W_feat_in: Num of feature input if mode contains 'f'. Used to calculate the first layer input dim.
                    By default 256
            act_cfg: cfg obj for selecting activation. None for relu.
                     For surface modeling, usually use SoftPlus(beta=100)
            weight_norm: If weight_norm, do extra weight_normalization for linear layer. By default False.
                         Nerf do not use it. Only for surface modeling methods(idr/neus/volsdf/unisurf)
        """
        super(VolRadianceNet, self).__init__()

        self.mode = mode
        assert len(mode) > 0 and all([m in 'pvnf' for m in mode]), 'Invalid mode only pvnf allowed...'

        # volume params
        self.n_grid = n_grid

        # nn params
        self.use_nn = use_nn
        self.W = W
        self.D = D
        self.W_feat_in = W_feat_in
        self.weight_norm = weight_norm

        self.dtype = dtype

        # set volume.
        self.volume = Volume(n_grid=n_grid, origin=origin, side=side, xlen=xlen, ylen=ylen, zlen=zlen, dtype=dtype)
        self.n_grid_pts = (n_grid + 1)**3
        self.n_voxel = n_grid**3

        # set grid_pts param for geo value and feature, in (n_grid+1)^3 dim, for volume
        self.grid_value_param = None
        self.layers = None
        self.embed_fn_pts, self.embed_fn_view = None, None
        self.init_input_dim = None

        # We can either use network or directly use volume
        if not use_nn:
            self.grid_value_param = nn.Parameter(torch.zeros(((n_grid + 1)**3, 3), dtype=dtype))
            nn.init.normal_(self.grid_value_param, 0.5, 0.1)  # rgb norm near 0.5
            self.grid_value_param.data = self.grid_value_param.data.clip(0.0, 1.0)  # force in range
        else:
            self.embed_fn_pts, self.embed_fn_view, self.init_input_dim, self.layers = self.setup_network(
                W, D, W_feat_in, act_cfg, input_ch_pts, input_ch_view, embed_freq_pts, embed_freq_view, *args, **kwargs
            )

    def get_volume(self):
        """Get the dense volume"""
        return self.volume

    def setup_network(
        self, W, D, W_feat_in, act_cfg, input_ch_pts, input_ch_view, embed_freq_pts, embed_freq_view, *args, **kwargs
    ):
        """Set up the nn network"""
        layers = []
        embed_fn_pts, embed_fn_view = None, None

        init_input_dim = 0
        # embedding for pts and view, calculate input shape
        if 'p' in self.mode:
            embed_fn_pts = Embedder(input_ch_pts, embed_freq_pts, *args, **kwargs)
            embed_pts_dim = embed_fn_pts.get_output_dim()
            init_input_dim += embed_pts_dim
        if 'v' in self.mode:
            embed_fn_view = Embedder(input_ch_view, embed_freq_view, *args, **kwargs)
            embed_view_dim = embed_fn_view.get_output_dim()
            init_input_dim += embed_view_dim
        if 'n' in self.mode:
            init_input_dim += 3
        if 'f' in self.mode and W_feat_in > 0:
            init_input_dim += W_feat_in

        for i in range(D + 1):
            in_dim = init_input_dim if i == 0 else W
            out_dim = W if i != D else 3

            if i != D:
                layer = DenseLayer(in_dim, out_dim, activation=get_activation(act_cfg))
            else:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())

            # extra weight normalization
            if self.weight_norm:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        layers = nn.ModuleList(layers)

        return embed_fn_pts, embed_fn_view, init_input_dim, layers

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor, normals: torch.Tensor, geo_feat: torch.Tensor):
        """
        Args:
            any of x/view_dir/normals/geo_feat are optional, based on mode
            x: torch.tensor (B, input_ch_pts)
            view_dirs: (B, input_ch_view)
            normals: (B, 3)
            geo_feat: (B, W_feat_in)

        Returns:
            out: tensor in shape (B, 3) for radiance value(rgb).
        """
        if self.use_nn:
            out = self.forward_nn(x, view_dirs, normals, geo_feat)
        else:
            out = self.forward_volume(x, view_dirs)

        return out

    def forward_nn(self, x: torch.Tensor, view_dirs: torch.Tensor, normals: torch.Tensor, geo_feat: torch.Tensor):
        """Forward use nn"""
        assert self.layers is not None, 'You do not init layer for processing'

        inputs = []
        if 'p' in self.mode:
            x_embed = self.embed_fn_pts(x)  # input_ch_pts -> embed_pts_dim
            inputs.append(x_embed)
        if 'v' in self.mode:
            view_embed = self.embed_fn_view(view_dirs)  # input_ch_view -> embed_view_dim
            inputs.append(view_embed)
        if 'n' in self.mode:
            inputs.append(normals)
        if 'f' in self.mode:
            inputs.append(geo_feat)

        out = torch.cat(inputs, dim=-1)
        assert out.shape[-1] == self.init_input_dim, 'Shape not match'

        for i in range(self.D + 1):
            out = self.layers[i](out)

        return out

    def forward_volume(self, x: torch.Tensor, view_dirs: torch.Tensor):
        """Forward use volume. only takes xyz and view"""
        dtype = x.dtype
        device = x.device
        n_pts = x.shape[0]

        # get valid voxel idx from pts xyz
        voxel_idx, valid_idx = self.volume.get_voxel_idx_from_xyz(x)  # (B, 3), xyz index of volume
        assert voxel_idx.max() < self.n_grid, 'Voxel idx exceed boundary...'

        out = torch.zeros((n_pts, 3), dtype=dtype).to(device)  # (B, 3)

        if torch.any(valid_idx):  # any points in volume
            # get valid grid pts position
            grid_pts_valid = self.volume.get_grid_pts_by_voxel_idx(voxel_idx[valid_idx])  # (B_valid, 8, 3)
            # calculate weights to 8 grid_pts by inverse distance
            grid_pts_weights_valid = self.volume.cal_weights_to_grid_pts(x[valid_idx], grid_pts_valid)  # (B_valid, 8)

            # select rgb value by index
            out[valid_idx] = self.volume.interpolate(
                self.grid_value_param, grid_pts_weights_valid, voxel_idx[valid_idx]
            )  # (B_valid, 3)

        # TODO: use view when using sphere hamonics

        # force rgb in (0~1)
        out = out.clip(0.0, 1.0)

        return out
