# -*- coding: utf-8 -*-

import math

import numpy as np
import torch
import torch.nn as nn

from .activation import get_activation
from .embed import Embedder
from .linear import DenseLayer, SirenLayer


class GeoNet(nn.Module):
    """Geometry network. Input xyz coord, get geometry value like density, sdf, occpancy, etc
        ref: https://github.com/ventusff/neurecon/blob/main/models/base.py
    """

    def __init__(
        self,
        W=256,
        D=8,
        skips=[4],
        input_ch=3,
        embed_freq=6,
        W_feat=256,
        skip_reduce_output=False,
        norm_skip=False,
        act_cfg=None,
        geometric_init=True,
        radius_init=1.0,
        use_siren=False,
        weight_norm=False,
        *args,
        **kwargs
    ):
        """
        Args:
            W: mlp hidden layer size, by default 256
            D: num of hidden layers, by default 8
            skips: list of skip points to add input directly, by default [4] (at middle). Must in range [0, D-1]
                    For any skip layer, it is concat in [feature, embed] order.
            input_ch: input channel num, by default 3(xyz). It is the dim before embed.
            embed_freq: embedding freq. by default 6. (Nerf use 10)
                        output dim will be input_ch * (freq * 2 + 1). 0 means not embed.
            W_feat: Num of feature output. If <1, not output any feature. By default 256
            skip_reduce_output: If True, reduce output dim by embed_dim, else cat embed to hidden. By default False.
                               Nerf do not use it. Only for surface modeling methods(idr/neus/volsdf/unisurf)
            norm_skip: If True, when concat [h, input], will norm by sqrt(2). By default False.
                        Nerf do not use it. Only for surface modeling methods(idr/neus/volsdf/unisurf)
            act_cfg: cfg obj for selecting activation. None for relu.
                     For surface modeling, usually use SoftPlus(beta=100)
            geometric_init: If True, init params by using geometric rule. For DenseLayer only. Siren needs pretrain.
                            Nerf do not use it. Only for surface modeling methods(idr/neus/volsdf/unisurf)
                            ref: https://github.com/matanatz/SAL
            radius_init: radius of sphere used for geometric_init. Output sdf can be init like sphere. By fault 1.
                         Nerf do not use it. Only for surface modeling methods(idr/neus/volsdf/unisurf)
            use_siren: If True, use SirenLayer instead of DenseLayer with sine activation. By fault False.
            weight_norm: If weight_norm, do extra weight_normalization. By default False.
                         Nerf do not use it. Only for surface modeling methods(idr/neus/volsdf/unisurf)
        """
        super(GeoNet, self).__init__()
        self.W = W
        self.D = D
        self.W_feat = W_feat
        self.skips = skips
        self.is_pretrained = False
        self.radius_init = radius_init
        self.geometric_init = geometric_init
        self.use_siren = use_siren
        if use_siren:
            assert len(skips) == 0, 'do not use skips for siren'
        self.norm_skip = norm_skip
        self.embed_fn = Embedder(input_ch, embed_freq, *args, **kwargs)
        embed_dim = self.embed_fn.get_output_dim()

        layers = []
        for i in range(D + 1):
            # input dim for each fc
            if i == 0:
                in_dim = embed_dim
            elif not skip_reduce_output and i in skips:  # skip at current layer, add input
                in_dim = embed_dim + W
            else:
                in_dim = W

            # out dim for each fc
            if i == D:  # last layer:
                if W_feat > 0:
                    out_dim = 1 + W_feat
                else:
                    out_dim = 1
            elif skip_reduce_output and (i + 1) in skips:  # skip at next, reduce current output
                out_dim = W - embed_dim
            else:
                out_dim = W

            # select layers. Last layer will not have activation
            if i != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first=(i == 0))
                else:
                    layer = DenseLayer(in_dim, out_dim, activation=get_activation(act_cfg))
            else:
                layer = nn.Linear(in_dim, out_dim)

            # geo_init for denseLayer. For any layer inputs, it should be [feature, x, embed_x]
            if geometric_init and not use_siren:
                if i == D:  # last layer
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    nn.init.constant_(layer.bias[:1], -radius_init)  # bias only for geo value
                elif embed_freq > 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    if i == 0:  # first layer, [x, embed_x], do not init for embed_x
                        torch.nn.init.constant_(layer.weight[:, input_ch:], 0.0)
                        torch.nn.init.normal_(layer.weight[:, :input_ch], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    elif i in skips:  # skip layer, [feature, x, embed_x], do not init for embed_x
                        torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                        torch.nn.init.constant_(layer.weight[:, -(embed_dim - input_ch):], 0.0)
                    else:
                        nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                else:
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            # extra weight normalization
            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def pretrain_siren(self, n_iter=5000, lr=1e-4, thres=0.01, n_pts=5000):
        """Pretrain the siren params."""
        if self.geometric_init and self.use_siren and not self.is_pretrained:
            sample_radius = self.radius_init * 2.0  # larger sample sphere
            pretrain_siren(self, self.radius_init, sample_radius, n_iter, lr, thres, n_pts)
            self.is_pretrained = True

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: torch.tensor (B, input_ch)

        Returns:
            tensor in shape (B, 1) for geometric value(sdf, sigma, occ).
            tensor in shape (B, W_feat) if W_feat > 0. None if W_feat <= 0
        """
        x = self.embed_fn(x)  # input_ch -> embed_dim
        out = x

        for i in range(self.D + 1):
            if i in self.skips:
                out = torch.cat([out, x], dim=-1)  # cat at last
                if self.norm_skip:
                    out = out / math.sqrt(2)
            out = self.layers[i](out)

        if self.W_feat <= 0:  # (B, 1), None
            return out, None
        else:  # (B, 1), (B, W_feat)
            return out[:, 0].unsqueeze(-1), out[:, 1:]

    def forward_geo_value(self, x: torch.Tensor):
        """Only get geometry value like sigma/sdf/occ"""
        return self.forward(x)[0]

    def forward_with_grad(self, x: torch.Tensor):
        """Get the grad of geo_value wrt input x. It could be the normal on surface"""
        with torch.enable_grad():
            x = x.requires_grad_(True)
            geo_value, h = self.forward(x)
            grad = torch.autograd.grad(
                outputs=geo_value,
                inputs=x,
                grad_outputs=torch.ones_like(geo_value, requires_grad=False, device=x.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

        return geo_value, h, grad


def pretrain_siren(model, radius_init, sample_radius, n_iter=5000, lr=1e-4, thres=0.01, n_pts=5000):
    """Pretrain the siren params. Radius_init is not allowed to be too large
    If you init larger sphere, you need to sample more pts to it for better result

    Args:
        model: implicit geometry model to process the pts
        radius_init: radius of the inner sphere
        sample_radius: radius of the sampling sphere space
        n_iter: num of total iteration. By default 5000.
        lr: learning rate. By default 1e-4(using adam)
        thres: thres to stop the training, should be proportional to sphere radius. By default 0.01.
        n_pts: num of pts sample. By default 5k for radius in 1.0.
    """
    assert radius_init <= 5.0, 'To large sphere, does not accept'
    assert radius_init < sample_radius, 'Sample space does not fully cover the sphere'

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l1loss = nn.L1Loss(reduction='mean')
    for _ in range(n_iter):
        pts = torch.empty([n_pts, 3], dtype=dtype).uniform_(-sample_radius, sample_radius).to(device)
        sdf_gt = pts.norm(dim=-1) - radius_init  # inside -/outside +
        sdf_pred = model(pts)[0]
        loss = l1loss(sdf_pred, sdf_gt[:, None])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < thres:
            break


class RadianceNet(nn.Module):
    """Radiance network. Input view direction(optional: xyz dir, feature, norm, etc), get rgb color.
        ref: https://github.com/ventusff/neurecon/blob/main/models/base.py
    """

    def __init__(
        self,
        mode='vf',
        W=256,
        D=8,
        input_ch_pts=3,
        input_ch_view=3,
        embed_freq_pts=6,
        embed_freq_view=4,
        W_feat_in=256,
        act_cfg=None,
        use_siren=False,
        weight_norm=False,
        *args,
        **kwargs
    ):
        """
        Args:
            mode: 'p':points(xyz) - 'v':view_dirs - 'n':normals - 'f'-geo_feat
                  Should be a str combining all used inputs. By default 'vf', use view_dir and geo_feat like nerf.
            W: mlp hidden layer size, by default 256
            D: num of hidden layers, by default 8
            input_ch_pts: input channel num for points, by default 3(xyz). It is the dim before embed.
            input_ch_view: input channel num for view_dirs, by default 3. It is the dim before embed.
            embed_freq_pts: embedding freq for pts. by default 6.
                            output dim will be input_ch * (freq * 2 + 1). 0 means not embed.
            embed_freq_view: embedding freq for view_dir. by default 4.
                            output dim will be input_ch * (freq * 2 + 1). 0 means not embed.
            W_feat: Num of feature input if mode contains 'f'. Used to calculate the first layer input dim.
                    By default 256
            act_cfg: cfg obj for selecting activation. None for relu.
                     For surface modeling, usually use SoftPlus(beta=100)
            use_siren: If True, use SirenLayer instead of DenseLayer with sine activation. By fault False.
            weight_norm: If weight_norm, do extra weight_normalization. By default False.
                         Nerf do not use it. Only for surface modeling methods(idr/neus/volsdf/unisurf)
        """
        super(RadianceNet, self).__init__()
        self.mode = mode
        assert len(mode) > 0 and all([m in 'pvnf' for m in mode]), 'Invalid mode only pvnf allowed...'
        self.W = W
        self.D = D
        self.W_feat_in = W_feat_in

        self.init_input_dim = 0
        # embedding for pts and view, calculate input shape
        if 'p' in mode:
            self.embed_fn_pts = Embedder(input_ch_pts, embed_freq_pts, *args, **kwargs)
            embed_pts_dim = self.embed_fn_pts.get_output_dim()
            self.init_input_dim += embed_pts_dim
        if 'v' in mode:
            self.embed_fn_view = Embedder(input_ch_view, embed_freq_view, *args, **kwargs)
            embed_view_dim = self.embed_fn_view.get_output_dim()
            self.init_input_dim += embed_view_dim
        if 'n' in mode:
            self.init_input_dim += 3
        if 'f' in mode and W_feat_in > 0:
            self.init_input_dim += W_feat_in

        layers = []
        for i in range(D + 1):
            # input dim for each fc
            if i == 0:
                in_dim = self.init_input_dim
            else:
                in_dim = W

            # out dim for each fc
            if i == D:  # last layer: rgb
                out_dim = 3
            else:
                out_dim = W

            # select layers. Last layer has sigmoid
            if i != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first=(i == 0))
                else:
                    layer = DenseLayer(in_dim, out_dim, activation=get_activation(act_cfg))
            else:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())

            # extra weight normalization
            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor, normals: torch.Tensor, geo_feat: torch.Tensor):
        """
        Args:
            any of x/view_dir/normals/geo_feat are optional, based on mode
            x: torch.tensor (B, input_ch_pts)
            view_dirs: (B, input_ch_view)
            normals: (B, 3)
            geo_feat: (B, W_feat_in)

        Returns:
            tensor in shape (B, 3) for radiance value(rgb).
        """
        inputs = []
        if 'p' in self.mode:
            x = self.embed_fn_pts(x)  # input_ch_pts -> embed_pts_dim
            inputs.append(x)
        if 'v' in self.mode:
            view_dirs = self.embed_fn_view(view_dirs)  # input_ch_view -> embed_view_dim
            inputs.append(view_dirs)
        if 'n' in self.mode:
            inputs.append(normals)
        if 'f' in self.mode:
            inputs.append(geo_feat)

        out = torch.cat(inputs, dim=-1)
        assert out.shape[-1] == self.init_input_dim, 'Shape not match'

        for i in range(self.D + 1):
            out = self.layers[i](out)

        return out
