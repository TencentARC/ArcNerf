# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from simplenerf.geometry.transformation import normalize


class Embedder(nn.Module):
    """
        Embedding module. Embed inputs into higher dimensions.
        For example, x = sin(2**N * x) or sin(N * x) for N in range(0, 10)
        ref: https://github.com/ventusff/neurecon/blob/main/models/base.py
    """

    def __init__(
        self,
        input_dim,
        n_freqs,
        periodic_fns=(torch.sin, torch.cos),
        *args,
        **kwargs
    ):
        """
        Args:
            input_dim: dimension of input to be embedded. For example, xyz is dim=3
            n_freqs: number of frequency bands. If 0, will not encode the inputs.
            periodic_fns: a list of periodic functions used to embed input. By default is (sin, cos)

        Returns:
            Embedded inputs with shape:
                (inputs_dim * len(periodic_fns) * N_freq + include_input * inputs_dim)
            For example, inputs_dim = 3, using (sin, cos) encoding, N_freq = 10, include_input, will results at
                3 * 2 * 10 + 3 = 63 output shape.
        """
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.periodic_fns = periodic_fns

        # get output dim
        self.out_dim = self.input_dim * (1 + n_freqs * len(self.periodic_fns))

        self.freq_bands = 2.**torch.linspace(0., n_freqs - 1, n_freqs)

    def get_output_dim(self):
        """Get output dim"""
        return self.out_dim

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: tensor of shape [B, input_dim]

        Returns:
            embed_x: tensor of shape [B, out_dim]
        """
        assert (x.shape[-1] == self.input_dim), 'Input shape should be (B, {})'.format(self.input_dim)

        embed_x = [x]

        for freq in self.freq_bands:
            for fn in self.periodic_fns:
                embed_x.append(fn(x * freq))

        embed_x = torch.cat(embed_x, dim=-1)

        return embed_x


class GeoNet(nn.Module):
    """Geometry network with linear network implementation.
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
        """
        super(GeoNet, self).__init__()
        self.W, self.D, self.W_feat = W, D, W_feat
        self.skips = skips
        self.embed_fn = Embedder(input_ch, embed_freq)
        embed_dim = self.embed_fn.get_output_dim()

        layers = []
        for i in range(D):
            # input dim for each fc
            if i == 0:
                in_dim = embed_dim
            elif i > 0 and (i-1) in skips:  # skip at current layer, add input
                in_dim = embed_dim + W
            else:
                in_dim = W

            layer = nn.Linear(in_dim, W)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        # final layer
        out_dim = 1 + W_feat if W_feat > 0 else 1
        self.final_layer = nn.Linear(W, out_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: torch.tensor (B, input_ch)

        Returns:
            out: tensor in shape (B, 1) for geometric value(sdf, sigma, occ).
            out_feat: tensor in shape (B, W_feat) if W_feat > 0. None if W_feat <= 0
        """
        x_embed = self.embed_fn(x)  # input_ch -> embed_dim
        out = x_embed

        for i in range(self.D):
            out = self.layers[i](out)
            out = F.relu(out)
            if i in self.skips:
                out = torch.cat([out, x_embed], dim=-1)  # cat at last

        out = self.final_layer(out)

        if self.W_feat <= 0:  # (B, 1), None
            return out, None
        else:  # (B, 1), (B, W_feat)
            return out[:, 0].unsqueeze(-1), out[:, 1:]


class RadianceNet(nn.Module):
    """Radiance network with linear network implementation.
        ref: https://github.com/ventusff/neurecon/blob/main/models/base.py
    """

    def __init__(
        self,
        W=256,
        D=8,
        input_ch_view=3,
        embed_freq_view=4,
        W_feat_in=256,
    ):
        """
        Args:
            W: mlp hidden layer size, by default 256
            D: num of hidden layers, by default 8
            input_ch_view: input channel num for view_dirs, by default 3. It is the dim before embed.
            embed_freq_view: embedding freq for view_dir. by default 4.
                            output dim will be input_ch * (freq * 2 + 1). 0 means not embed.
            W_feat_in: Num of feature input if mode contains 'f'. Used to calculate the first layer input dim.
                    By default 256
        """
        super(RadianceNet, self).__init__()
        self.W, self.D, self.W_feat_in = W, D, W_feat_in

        # view embed
        self.embed_fn_view = Embedder(input_ch_view, embed_freq_view)

        self.init_input_dim = self.embed_fn_view.get_output_dim() + W_feat_in

        layers = []
        for i in range(D):
            in_dim = self.init_input_dim if i == 0 else W
            layer = nn.Linear(in_dim, W)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        # final layer
        self.final_layer = nn.Linear(W, 3)

    def forward(self, rays_d: torch.Tensor, geo_feat: torch.Tensor):
        """
        Args:
            rays_d: (B, input_ch_view), this is not normed direction
            geo_feat: (B, W_feat_in)

        Returns:
            out: tensor in shape (B, 3) for radiance value(rgb).
        """
        view_dirs = normalize(rays_d)  # norm the rays_d as view direction
        view_embed = self.embed_fn_view(view_dirs)  # input_ch_view -> embed_view_dim

        inputs = [view_embed, geo_feat]
        out = torch.cat(inputs, dim=-1)
        assert out.shape[-1] == self.init_input_dim, 'Shape not match'

        for i in range(self.D):
            out = self.layers[i](out)
            out = F.relu(out)

        out = self.final_layer(out)
        out = torch.sigmoid(out)

        return out
