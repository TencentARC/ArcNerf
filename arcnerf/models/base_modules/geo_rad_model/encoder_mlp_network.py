# -*- coding: utf-8 -*-

import torch

from .base_network import BaseGeoNet, BaseRadianceNet
from arcnerf.geometry.transformation import normalize
from arcnerf.models.base_modules import build_encoder, get_activation


class EncoderMLPGeoNet(BaseGeoNet):
    """Geometry network with Encoder+MLP strcuture
     Input xyz coord, get geometry value like density, sdf, occupancy, etc
    """

    def __init__(self, W_feat=256, out_act_cfg=None):
        super(EncoderMLPGeoNet, self).__init__()
        self.W_feat = W_feat
        self.embed_dim = 0
        self.embed_fn = None

        # out_act
        self.out_act = None
        if out_act_cfg is not None:
            self.out_act = get_activation(cfg=out_act_cfg)

    def build_encoder(self, encoder):
        """Build encoder for geometry"""
        self.embed_fn, input_ch, embed_freq = build_encoder(encoder)
        self.embed_dim = self.embed_fn.get_output_dim()

        return input_ch, embed_freq

    def build_mlp(self, **kwargs):
        """Build the mlp network"""
        raise NotImplementedError('You must implement this function')

    def handle_output(self, out):
        """Separate the output geo/feat from mlp"""
        out_geo, out_feat = None, None
        if self.W_feat <= 0:  # (B, 1), None
            out_geo = out
        else:  # (B, 1), (B, W_feat)
            out_geo = out[:, 0].unsqueeze(-1)
            out_feat = out[:, 1:]

        if self.out_act is not None:
            out_geo = self.out_act(out_geo)

        return out_geo, out_feat


class EncoderMLPRadainceNet(BaseRadianceNet):
    """Radiance network with Encoder+MLP strcuture
     Input view direction(optional: xyz dir, feature, norm, etc), get rgb color.
    """

    def __init__(self, mode='vf'):
        """
        Args:
            mode: 'p':points(xyz) - 'v':view_dirs - 'n':normals - 'f'-geo_feat
                  Should be a str combining all used inputs. By default 'vf', use view_dir and geo_feat like nerf.
        """
        super(EncoderMLPRadainceNet, self).__init__()
        self.mode = mode
        assert len(mode) > 0 and all([m in 'pvnf' for m in mode]), 'Invalid mode only pvnf allowed...'

        self.init_input_dim = 0
        self.embed_fn_pts = None
        self.embed_fn_view = None

    def build_encoder(self, encoder, W_feat_in):
        """Build encoder for radiance"""
        # embedding for pts and view, calculate input shape
        if 'p' in self.mode:
            self.embed_fn_pts, _, _ = build_encoder(encoder.pts if encoder is not None else None)
            embed_pts_dim = self.embed_fn_pts.get_output_dim()
            self.init_input_dim += embed_pts_dim
        if 'v' in self.mode:
            self.embed_fn_view, _, _ = build_encoder(encoder.view if encoder is not None else None)
            embed_view_dim = self.embed_fn_view.get_output_dim()
            self.init_input_dim += embed_view_dim
        if 'n' in self.mode:
            self.init_input_dim += 3
        if 'f' in self.mode and W_feat_in > 0:
            self.init_input_dim += W_feat_in

    def build_mlp(self, **kwargs):
        """Build the mlp network"""
        raise NotImplementedError('You must implement this function')

    def fuse_radiance_inputs(self, x, view_dirs, normals, geo_feat):
        """Fuse inputs for radiance with embedding"""
        inputs = []
        if 'p' in self.mode:
            x_embed = self.embed_fn_pts(x)  # input_ch_pts -> embed_pts_dim
            inputs.append(x_embed)
        if 'v' in self.mode:  # always normalize view_dirs
            view_embed = self.embed_fn_view(normalize(view_dirs))  # input_ch_view -> embed_view_dim
            inputs.append(view_embed)
        if 'n' in self.mode:
            inputs.append(normals)
        if 'f' in self.mode:
            inputs.append(geo_feat)

        out = torch.cat(inputs, dim=-1)
        assert out.shape[-1] == self.init_input_dim, 'Shape not match'

        return out
