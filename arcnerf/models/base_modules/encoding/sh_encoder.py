# -*- coding: utf-8 -*-

import warnings

import torch
import torch.nn as nn

from . import ENCODER_REGISTRY

# import tcnn encoder
try:
    import tinycudann as tcnn
    TCNN_BACKEND_AVAILABLE = True
except ImportError:
    TCNN_BACKEND_AVAILABLE = False
    warnings.warn('TCNN not import correctly...Possibly not build yet...')


@ENCODER_REGISTRY.register()
class SHEmbedder(nn.Module):
    """Spherical Harmonics Embedder in torch. Embed view dir into higher dimensions.
    Detail introduction is at: https://en.wikipedia.org/wiki/Spherical_harmonics
    ref: https://github.com/yashbhalgat/HashNeRF-pytorch/blob/main/hash_encoding.py
    This can be only used for xyz direction, but not position
    """

    def __init__(self, input_dim=3, n_freqs=4, include_input=True, backend=None, *args, **kwargs):
        """
        Args:
            input_dim: dimension of input to be embedded. Must be 3(direction)
            n_freqs: num of degree for embedding.
            include_input: if True, raw input is included in the embedding. Appear at beginning. By default is True.
            backend: which backend to use. By default None, use pure torch version.

        Returns:
            Embedded inputs with shape:
                n_freqs**2 + include_inputs * input_dim
        """
        super(SHEmbedder, self).__init__()

        assert input_dim == 3, 'SHEmbedder should has input_dim==3...'
        assert 1 <= n_freqs <= 5, 'Should have degree 1~5 for encoding...'
        self.input_dim = input_dim
        self.n_freqs = n_freqs
        self.include_input = include_input

        # backend
        if backend is None:
            backend = 'torch'
        assert backend in ['torch', 'tcnn'], 'Invalid backend used, only torch/tcnn allowed'
        self.backend = backend

        # set up tcnn backend if required
        if self.backend == 'tcnn' and TCNN_BACKEND_AVAILABLE:
            self.sh_encode_tcnn = tcnn.Encoding(
                n_input_dims=input_dim, encoding_config={
                    'otype': 'SphericalHarmonics',
                    'degree': n_freqs
                }
            )

        self.out_dim = n_freqs**2 + include_input * input_dim

    def get_output_dim(self):
        """Get output dim"""
        return self.out_dim

    @staticmethod
    def get_factors(degree):
        """Get factors for sh multiplication.
        Check this: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics

        Args:
            degree: num of degree to expand, generally between 1~5

        Returns:
            freq_factors: list of list. Len of list = degree, sum of all list num = degree**2
        """
        sh_factors = [
            [0.28209479177387814],  # l=1, 1, cumsum=1
            [-0.4886025119029199, 0.4886025119029199, -0.4886025119029199],  # l=2, 3, cumsum=4
            [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
             0.5462742152960396],  # l=3, 5, cumsum=9
            [
                -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
                1.445305721320277, -0.5900435899266435
            ],  # l=4, 7, cumsum=16
            [
                2.5033429417967046, -1.7701307697799304, 0.9461746957575601, -0.6690465435572892, 0.10578554691520431,
                -0.6690465435572892, 0.47308734787878004, -1.7701307697799304, 0.6258357354491761
            ]  # l=5, 9, cumsum=25
        ]

        return sh_factors[:degree]

    def forward(self, xyz: torch.Tensor):
        """
        Args:
            xyz: tensor of shape (B, 3), xyz direction, normalized

        Returns:
            out: tensor of shape (B, out_dim=n_freqs**2 + include_inputs * input_dim)
        """
        assert len(xyz.shape) == 2 and xyz.shape[-1] == 3, 'Must be (B, 3) direction'

        out = []
        if self.include_input:
            out.append(xyz)  # (B, 3)

        # norm all the dir (-1, 1) -> (0, 1) for spherical harmonic
        xyz_norm = (xyz + 1) / 2.0
        if self.backend == 'tcnn' and TCNN_BACKEND_AVAILABLE:
            sh_embed = self.sh_encode_tcnn(xyz_norm)
        else:
            sh_embed = self.sh_encode_torch(xyz_norm)
        out.append(sh_embed)  # (B, n_freqs**2)

        return torch.cat(out, dim=-1)  # (B, out_dim)

    def sh_encode_torch(self, xyz: torch.Tensor):
        """Embed xyz direction into n_freqs**2 feature

        Args:
            xyz: tensor of shape (B, 3), xyz direction, normalized

        Returns:
            out: tensor of shape (B, out_dim=n_freqs**2)
        """
        dtype = xyz.dtype
        device = xyz.device

        out = []

        freq_factors = self.get_factors(self.n_freqs)
        x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]  # (B, 1)
        xx, yy, zz = x**2, y**2, z**2
        xy, yz, xz = x * y, y * z, x * z

        # hardcode the operation
        fac_0 = torch.tensor([freq_factors[0][0]], dtype=dtype, device=device).unsqueeze(0)
        fac_0 = torch.repeat_interleave(fac_0, xyz.shape[0], dim=0)  # (B, 1)
        out.append(fac_0)
        if self.n_freqs <= 1:
            return torch.cat(out, dim=-1)  # (B, 1)

        out.append(freq_factors[1][0] * y)
        out.append(freq_factors[1][1] * z)
        out.append(freq_factors[1][2] * x)
        if self.n_freqs <= 2:
            return torch.cat(out, dim=-1)  # (B, 4)

        out.append(freq_factors[2][0] * xy)
        out.append(freq_factors[2][1] * yz)
        out.append(freq_factors[2][2] * (3.0 * zz - 1.0))
        out.append(freq_factors[2][3] * xz)
        out.append(freq_factors[2][4] * (xx - yy))
        if self.n_freqs <= 3:
            return torch.cat(out, dim=-1)  # (B, 9)

        out.append(freq_factors[3][0] * y * (3.0 * xx - yy))
        out.append(freq_factors[3][1] * xy * z)
        out.append(freq_factors[3][2] * y * (5.0 * zz - 1.0))
        out.append(freq_factors[3][3] * z * (5.0 * zz - 3.0))
        out.append(freq_factors[3][4] * x * (5.0 * zz - 1.0))
        out.append(freq_factors[3][5] * z * (xx - yy))
        out.append(freq_factors[3][6] * x * (xx - 3.0 * yy))
        if self.n_freqs <= 4:
            return torch.cat(out, dim=-1)  # (B, 16)

        out.append(freq_factors[4][0] * xy * (xx - yy))
        out.append(freq_factors[4][1] * yz * (3.0 * xx - yy))
        out.append(freq_factors[4][2] * xy * (7.0 * zz - 1.0))
        out.append(freq_factors[4][3] * yz * (7.0 * zz - 3.0))
        out.append(freq_factors[4][4] * (zz * (35.0 * zz - 30.0) + 3.0))
        out.append(freq_factors[4][5] * xz * (7.0 * zz - 3.0))
        out.append(freq_factors[4][6] * (xx - yy) * (7.0 * zz - 1.0))
        out.append(freq_factors[4][7] * xz * (xx - 3.0 * yy))
        out.append(freq_factors[4][8] * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)))

        return torch.cat(out, dim=-1)  # (B, 25)
