# -*- coding: utf-8 -*-

import math
import warnings

import torch
import torch.nn as nn

from . import ENCODER_REGISTRY
from arcnerf.geometry.volume import Volume
# import customized hashgrid encode
try:
    from arcnerf.ops import HashGridEncode
    CUDA_BACKEND_AVAILABLE = True
except ImportError:
    CUDA_BACKEND_AVAILABLE = False
    warnings.warn('HashGridEncode not import correctly...Possibly not build yet...')
# import tcnn encoder
try:
    import tinycudann as tcnn
    TCNN_BACKEND_AVAILABLE = True
except ImportError:
    TCNN_BACKEND_AVAILABLE = False
    warnings.warn('TCNN not import correctly...Possibly not build yet...')


@ENCODER_REGISTRY.register()
class HashGridEmbedder(nn.Module):
    """The multi-res hash-grid embedder introduced in instant-ngp
    ref: https://github.com/NVlabs/tiny-cuda-nn / https://github.com/ashawkey/torch-ngp
    """

    def __init__(
        self,
        input_dim=3,
        n_levels=16,
        n_feat_per_entry=2,
        hashmap_size=19,
        base_res=16,
        max_res=512,
        origin=(0, 0, 0),
        side=1.5,
        xlen=None,
        ylen=None,
        zlen=None,
        dtype=torch.float32,
        include_input=True,
        backend=None,
        *args,
        **kwargs
    ):
        """
        Args:
            input_dim: dimension of input to be embedded. Must be 3(direction)
            n_levels: num of levels of embedding(L), by default 16
            n_feat_per_entry: num of feat for each entry in hashmap(F), by default 2
            hashmap_size: 2-based hashmap size for each level(T), by default 19 (2**19 table)
            base_res: base resolution. By default 16.
            max_res: max res. By default 512.
                The scale factor is exp(ln(max_res/base_res) / (L-1)).
                Each level res is base * (scale ** L).
            The following are for volume:
                origin: origin point(centroid of cube), a tuple of 3
                side: each side len, if None, use xyzlen. If exist, use side only. By default 1.5.
                xlen: len of x dim, if None use side
                ylen: len of y dim, if None use side
                zlen: len of z dim, if None use side
                dtype: dtype of params. By default is torch.float32
            include_input: if True, raw input is included in the embedding. Appear at beginning. By default is True
            backend: which backend to use. By default None, use pure torch version.

        Returns:
            Embedded inputs with shape:
                L * F (each level, get the output from hashtable and concat) + include_inputs * input_dim
        """
        super(HashGridEmbedder, self).__init__()

        assert input_dim == 3, 'HashGridEmbedder should has input_dim==3...'
        self.input_dim = input_dim
        self.include_input = include_input

        # eq.3 in paper, each level multiply this scale, at most will be max_res
        self.base_res = base_res
        self.max_res = max_res
        self.per_level_scale = torch.exp((torch.log(torch.tensor(max_res / base_res))) / (float(n_levels) - 1))

        # embedding for each level with initialization
        self.hashmap_size = 2**hashmap_size  # T
        self.n_feat_per_entry = n_feat_per_entry  # F
        self.n_levels = n_levels  # L
        init_emb = not (backend == 'tcnn' and TCNN_BACKEND_AVAILABLE)
        self.embeddings, self.n_total_embed, self.offsets, self.resolutions = self.init_embeddings(init_emb)

        # set volume with base res
        self.volume = Volume(
            n_grid=self.base_res, origin=origin, side=side, xlen=xlen, ylen=ylen, zlen=zlen, dtype=dtype
        )

        # register params
        self.register()

        # backend
        if backend is None:
            backend = 'torch'
        assert backend in ['torch', 'cuda', 'tcnn'], 'Invalid backend used, only torch/cuda/tcnn allowed'
        self.backend = backend

        # set up cuda backend
        if self.backend == 'cuda' and CUDA_BACKEND_AVAILABLE:
            self.hashgrid_encode_cuda = HashGridEncode(self.n_levels, self.n_feat_per_entry)
        elif self.backend == 'tcnn' and TCNN_BACKEND_AVAILABLE:
            self.hashgrid_encode_tcnn = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                    'otype': 'HashGrid',
                    'n_levels': self.n_levels,
                    'n_features_per_level': self.n_feat_per_entry,
                    'log2_hashmap_size': hashmap_size,
                    'base_resolution': self.base_res,
                    'per_level_scale': float(self.per_level_scale),
                },
                dtype=dtype
            )

        self.out_dim = n_levels * n_feat_per_entry + include_input * input_dim  # L * F + 3

    def get_output_dim(self):
        """Get output dim"""
        return self.out_dim

    def register(self):
        """To make those tensor into the same device"""
        self.register_buffer('min_xyz', self.volume.get_range()[:, 0])  # (3,)
        self.register_buffer('max_xyz', self.volume.get_range()[:, 1])  # (3,)
        self.register_buffer('t_offsets', torch.tensor(self.offsets, dtype=torch.int))  # (L+1,)
        self.register_buffer('t_resolutions', torch.tensor(self.resolutions, dtype=torch.int))  # (L,)

    def init_embeddings(self, init_emb, std=1e-4):
        """Init embedding. To save memory, at lower level, do not init large embeddings

        Args:
            init_emb: whether you need to set up the real embedding tensor. False for tcnn backend
            std: for embedding init, by default 1e-4

        Returns:
            embeddings: embedding vec in shape (n_total_embed, F)
            n_total_embed: the overall embed size, less than (T * L)
            offsets: a list of offset of each level, len is L+1
            resolutions: a list of resolution at each level, len is L
        """
        offsets = []
        resolutions = []
        n_total_embed = 0

        for i in range(self.n_levels):
            offsets.append(n_total_embed)
            cur_res = math.floor(self.base_res * self.per_level_scale**i)
            resolutions.append(cur_res)
            n_embed_per_level = min(self.hashmap_size, (cur_res + 1)**3)  # save memory for low res
            n_total_embed += n_embed_per_level

        offsets.append(n_total_embed)

        embeddings = None
        if init_emb:
            embeddings = nn.Parameter(torch.empty(n_total_embed, self.n_feat_per_entry))
            nn.init.uniform_(embeddings, -std, std)

        return embeddings, n_total_embed, offsets, resolutions

    def get_embeddings(self):
        """Get the embeddings data"""
        return self.embeddings.data

    def set_embeddings(self, data):
        """Set the embeddings data"""
        self.embeddings.data = data

    def forward(self, xyz: torch.Tensor):
        """
        Args:
            xyz: tensor of shape (B, 3), xyz position. You should make sure the xyz in the whole bbox.

        Returns:
            out: tensor of shape (B, out_dim=T*F + include_inputs * input_dim)
        """
        assert len(xyz.shape) == 2 and xyz.shape[-1] == 3, 'Must be (B, 3) tensor'

        out = []
        if self.include_input:
            out.append(xyz)  # (B, 3)

        if self.backend == 'cuda' and CUDA_BACKEND_AVAILABLE:
            hashgrid_embed = self.hashgrid_encode_cuda(
                xyz, self.embeddings, self.t_offsets, self.t_resolutions, self.min_xyz, self.max_xyz
            )
        elif self.backend == 'tcnn' and TCNN_BACKEND_AVAILABLE:
            norm_xyz = (xyz - self.min_xyz) / (self.max_xyz - self.min_xyz)  # to (0~1)
            hashgrid_embed = self.hashgrid_encode_tcnn(norm_xyz)
        else:
            hashgrid_embed = self.hashgrid_encode_torch(xyz)
        out.append(hashgrid_embed)  # (B, T*F)

        return torch.cat(out, dim=-1)  # (B, out_dim)

    def hashgrid_encode_torch(self, xyz: torch.Tensor):
        """Embed xyz position into T*F feature

        Args:
            xyz: tensor of shape (B, 3), xyz position

        Returns:
            out: tensor of shape (B, out_dim=L*F)
        """
        out = []
        empty_level_out = (xyz.clone() * 0.0)[:, :1].repeat(1, self.n_feat_per_entry)  # (B, F)

        for i, n_grid in enumerate(self.resolutions):
            out_level = empty_level_out.clone()  # (B, F)

            self.volume.set_n_grid(n_grid, reset_pts=False)  # reset n_grid, but not the grid_pts
            voxel_grid = self.volume.get_voxel_grid_info_from_xyz(xyz)

            # get used info
            valid_idx, grid_pts_idx_valid, grid_pts_weights_valid = voxel_grid[1], voxel_grid[2], voxel_grid[-1]

            if grid_pts_idx_valid is not None:
                # hash the index of the grid pts
                valid_grid_pts_hash_idx = self.fast_hash(grid_pts_idx_valid, self.offsets[i + 1] - self.offsets[i])
                valid_grid_pts_hash_idx += self.offsets[i]  # (B_valid, 8), in correct bin

                # embed on grid
                valid_embed = self.embeddings[valid_grid_pts_hash_idx.view(-1)]  # (B_valid*8, F)
                valid_embed = valid_embed.view(-1, 8, self.n_feat_per_entry)  # (B_valid, 8, F)

                # only update the embedding for bounding xyz
                out_level[valid_idx] = self.volume.interpolate_values_by_weights(
                    valid_embed, grid_pts_weights_valid
                )  # (B_valid, F)

            out.append(out_level)  # (B, F)

        out = torch.cat(out, dim=-1)  # (B, L*F)

        return out

    @staticmethod
    def fast_hash(idx: torch.Tensor, hashmap_size):
        """Hash the corner index

        Args:
            idx: (..., 3) index of xyz, long(int64) integer
            hashmap_size: size of the current hashmap

        Return:
            hash_index: (..., ) hash index, long(int64) integer
        """
        primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]  # at most 7

        hash_index = torch.zeros_like(idx[..., 0], dtype=torch.long)  # (B,)
        for i in range(idx.shape[-1]):
            hash_index ^= idx[..., i] * primes[i]

        return hash_index % hashmap_size
