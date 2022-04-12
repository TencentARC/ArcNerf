# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Embedder(nn.Module):
    """
        Embedding module. Embed inputs into higher dimensions.
        For example, x = sin(2**N * x) or sin(N * x) for N in range(0, 10)
        ref: https://github.com/ventusff/neurecon/blob/main/models/base.py
    """

    def __init__(self, input_dim, n_freqs, log_sampling=True, include_input=True, periodic_fns=(torch.sin, torch.cos)):
        """
        Args:
            input_dim: dimension of input to be embedded. For example, xyz is dim=3
            n_freqs: number of frequency bands. If 0, will not encode the inputs.
            log_sampling: if True, use log factor sin(2**N * x). Else use scale factor sin(N * x).
                      By default is True
            include_input: if True, raw input is included in the embedding. Appear at beginning. By default is True
            periodic_fns: a list of periodic functions used to embed input. By default is (sin, cos)

        Returns:
            Embedded inputs with shape:
                (inputs_dim * len(periodic_fns) * N_freq + include_input * inputs_dim)
            For example, inputs_dim = 3, using (sin, cos) encoding, N_freq = 10, include_input, will results at
                3 * 2 * 10 + 3 = 63 output shape.
        """
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        # get output dim
        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim
        self.out_dim += self.input_dim * n_freqs * len(self.periodic_fns)

        if n_freqs == 0 and include_input:  # inputs only
            self.freq_bands = []
        else:
            if log_sampling:
                self.freq_bands = 2.**torch.linspace(0., n_freqs - 1, n_freqs)
            else:
                self.freq_bands = torch.linspace(2.**0., 2.**(n_freqs - 1), n_freqs)

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

        embed_x = []
        if self.include_input:
            embed_x.append(x)

        for freq in self.freq_bands:
            for fn in self.periodic_fns:
                embed_x.append(fn(x * freq))

        if len(embed_x) > 1:
            embed_x = torch.cat(embed_x, dim=-1)
        else:
            embed_x = embed_x[0]

        return embed_x
