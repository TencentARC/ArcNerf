# -*- coding: utf-8 -*-

import warnings

import torch

from arcnerf.models.base_modules import MODULE_REGISTRY
from .encoder_mlp_network import EncoderMLPGeoNet, EncoderMLPRadainceNet
try:
    import tinycudann as tcnn
except ImportError:
    warnings.warn('TCNN Not build...')


@MODULE_REGISTRY.register()
class FusedMLPGeoNet(EncoderMLPGeoNet):
    """Geometry network with fused MLP From tinytcnn. You must install tcnn from:
        ref: https://github.com/NVlabs/tiny-cuda-nn
       By default tcnn runs out f16 tensor, You can cast them to float32 if necessary.
       Compared to linear_network_module, it is faster but lose flexibility(initialization, etc. )

       (FullyFusedMLP only processes fix num of neurons: 16, 21, 64, 128)
       No bias is used in the fc layer.
    """

    def __init__(
        self,
        W=128,
        D=8,
        encoder=None,
        W_feat=128,
        act_cfg=None,
        out_act_cfg=None,
        dtype=torch.float32,
        *args,
        **kwargs
    ):
        """
        Args:
            W: mlp hidden layer size, by default 128.
            D: num of hidden layers, by default 4
            skips: list of skip points to add input directly, by default [4] (at middle). Must in range [0, D-1]
                    For any skip layer, it is concat in [feature, embed] order.
            encoder: cfgs for encoder.
            W_feat: Num of feature output. If <1, not output any feature. By default 128
            act_cfg: cfg obj for selecting activation. None for relu.
            out_act_cfg: if not None, will perform an activation ops for output sigma.
            dtype: cast the output f16 to f32/64. By default use float32
        """
        super(FusedMLPGeoNet, self).__init__(W_feat=W_feat, out_act_cfg=out_act_cfg)
        self.W = W
        self.D = D
        self.dtype = dtype

        # build encoder
        _, _ = self.build_encoder(encoder)

        # build mlp
        self.layers = self.build_mlp(W_feat, act_cfg)

    def build_mlp(self, W_feat, act_cfg):
        """Return a fused mlp layer"""
        layers = \
            tcnn.Network(
                n_input_dims=self.embed_dim,
                n_output_dims=1 + W_feat if W_feat > 0 else 1,
                network_config={
                    'otype': 'FullyFusedMLP',
                    'activation': get_tcnn_activation_from_cfgs(act_cfg, 'ReLU'),
                    'output_activation': 'None',  # do not use default
                    'n_neurons': self.W,
                    'n_hidden_layers': self.D,
                }
            )

        return layers

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: torch.tensor (B, input_ch)

        Returns:
            out_geo: tensor in shape (B, 1) for geometric value(sdf, sigma, occ).
                     Perform activation if out_act is not None.
            out_feat: tensor in shape (B, W_feat) if W_feat > 0. None if W_feat <= 0
        """
        x_embed = self.embed_fn(x)  # input_ch -> embed_dim
        out = self.layers(x_embed).type(self.dtype)  # cast type

        # separate geo and feat
        out_geo, out_feat = self.handle_output(out)

        return out_geo, out_feat


@MODULE_REGISTRY.register()
class FusedMLPRadianceNet(EncoderMLPRadainceNet):
    """Radiance network with fused MLP From tinytcnn. You must install tcnn from:
        ref: https://github.com/NVlabs/tiny-cuda-nn
       By default tcnn runs out f16 tensor, You can cast them to float32 if necessary.
    """

    def __init__(
        self,
        mode='vf',
        W=128,
        D=8,
        encoder=None,
        W_feat_in=128,
        act_cfg=None,
        out_act_cfg=None,
        dtype=torch.float32,
        *args,
        **kwargs
    ):
        """
        Args:
            mode: 'p':points(xyz) - 'v':view_dirs - 'n':normals - 'f'-geo_feat
                  Should be a str combining all used inputs. By default 'vf', use view_dir and geo_feat like nerf.
            W: mlp hidden layer size, by default 128
            D: num of hidden layers, by default 8
            encoder: cfgs for encoder. Contains 'pts'/'view' fields for different input embedding.
            W_feat_in: Num of feature input if mode contains 'f'. Used to calculate the first layer input dim.
                    By default 128
            act_cfg: cfg obj for selecting activation. None for relu.
                     For surface modeling, usually use SoftPlus(beta=100)
            out_act_cfg: By default use 'Sigmoid' on rgb
            dtype: cast the output f16 to f32/64. By default use float32
        """
        super(FusedMLPRadianceNet, self).__init__(mode=mode)
        self.W = W
        self.D = D
        self.W_feat_in = W_feat_in
        self.dtype = dtype

        # build encoder
        self.build_encoder(encoder, W_feat_in)

        # build the fusedMLP
        self.layers = self.build_mlp(act_cfg, out_act_cfg)

    def build_mlp(self, act_cfg=None, out_act_cfg=None):
        """Return a fused mlp layer"""
        layers = \
            tcnn.Network(
                n_input_dims=self.init_input_dim,
                n_output_dims=3,
                network_config={
                    'otype': 'FullyFusedMLP',
                    'activation': get_tcnn_activation_from_cfgs(act_cfg, 'ReLU'),
                    'output_activation': get_tcnn_activation_from_cfgs(out_act_cfg, 'Sigmoid'),
                    'n_neurons': self.W,
                    'n_hidden_layers': self.D,
                }
            )

        return layers

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor, normals: torch.Tensor, geo_feat: torch.Tensor):
        """
        Args:
            any of x/view_dir/normals/geo_feat are optional, based on mode
            x: torch.tensor (B, input_ch_pts)
            view_dirs: (B, input_ch_view), may not be normalized
            normals: (B, 3)
            geo_feat: (B, W_feat_in)

        Returns:
            out: tensor in shape (B, 3) for radiance value(rgb).
        """
        out = self.fuse_radiance_inputs(x, view_dirs, normals, geo_feat)
        out = self.layers(out).type(self.dtype)  # cast type

        return out


def get_tcnn_activation_from_cfgs(cfg, default='None'):
    """Get the activation name from cfgs"""
    if cfg is None:
        return default

    if cfg.type.lower() == 'relu':
        return 'ReLU'
    elif cfg.type.lower() == 'exponential':
        return 'Exponential'
    elif cfg.type.lower() == 'sine':
        return 'Sine'
    elif cfg.type.lower() == 'sigmoid':
        return 'Sigmoid'
    elif cfg.type.lower() == 'squareplus':
        return 'Squareplus'
    elif cfg.type.lower() == 'softplus':
        return 'Softplus'
    else:
        raise NotImplementedError('No activation class {} in TinyCudaNN'.format(cfg.type))
