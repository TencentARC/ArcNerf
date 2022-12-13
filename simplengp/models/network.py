# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from common.utils.cfgs_utils import get_value_from_cfgs_field
from simplengp.ops.trunc_exp import trunc_exp
import tinycudann as tcnn


class NGPNetwork(nn.Module):
    """It is a hardcode class of ngp network with encoder + fused lp"""

    def __init__(self, cfgs, aabb_range):
        super(NGPNetwork, self).__init__()

        self.dtype = torch.float32

        # cfgs
        self.n_levels = get_value_from_cfgs_field(cfgs, 'n_levels', 16)
        self.n_feat_per_entry = get_value_from_cfgs_field(cfgs, 'n_feat_per_entry', 2)
        self.hashmap_size = get_value_from_cfgs_field(cfgs, 'hashmap_size', 19)
        self.base_res = get_value_from_cfgs_field(cfgs, 'base_res', 16)
        self.max_res = get_value_from_cfgs_field(cfgs, 'max_res', 2048)
        self.aabb_range = aabb_range

        # geo network
        self.xyz_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": self.n_levels,
                    "n_features_per_level": self.n_feat_per_entry,
                    "log2_hashmap_size": self.hashmap_size,
                    "base_resolution": self.base_res,
                    "per_level_scale": np.exp(np.log(self.max_res/self.base_res)/(self.n_levels-1)),
                },
            )

        self.xyz_net = \
            tcnn.Network(
                n_input_dims=self.n_levels * self.n_feat_per_entry,
                n_output_dims=16,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )

        # radiance network
        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        self.rgb_net = \
            tcnn.Network(
                n_input_dims=32,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",   # activation here but not in kernel
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )

    def forward_geo_value(self, x: torch.Tensor):
        """Only get geometry value like sigma/sdf/occ. In shape (B,) """
        # norm input
        x_norm = (x - self.aabb_range[0]) / (self.aabb_range[1] - self.aabb_range[0])

        xyz_embed = self.xyz_encoder(x_norm)  # (B, 32)
        xyz_out = self.xyz_net(xyz_embed)  # (B, 16)

        # apply activation
        sigma = trunc_exp(xyz_out[:, 0].type(self.dtype))

        return sigma

    def forward(self, x: torch.Tensor, dirs: torch.Tensor):
        """
        Args:
            x: pos, torch.tensor (B, 3)
            dirs: direction, torch.tensor (B, 3)

        Returns:
            sigma: tensor in shape (B,) for sigma
            radiance: tensor in shape (B, 3) for rgb
        """
        # norm input
        x_norm = (x - self.aabb_range[0]) / (self.aabb_range[1] - self.aabb_range[0])
        dirs_norm = (dirs + 1.0) / 2.0

        xyz_embed = self.xyz_encoder(x_norm)  # (B, 32)
        xyz_out = self.xyz_net(xyz_embed)  # (B, 16)

        dirs_embed = self.dir_encoder(dirs_norm)  # (B, 16)
        dirs_embed = torch.cat([xyz_out, dirs_embed], dim=-1)  # (B, 32)
        radiance = self.rgb_net(dirs_embed)  # (B, 3)

        # apply activation
        sigma = trunc_exp(xyz_out[:, 0].type(self.dtype))
        radiance = radiance.type(self.dtype)

        return sigma, radiance
