# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn

from . import ENCODER_REGISTRY
from arcnerf.geometry.volume import Volume
try:
    from arcnerf.ops import HashGridEncode
    CUDA_BACKEND_AVAILABLE = True
except ImportError:
    CUDA_BACKEND_AVAILABLE = False


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
        use_cuda_backend=False,
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
            use_cuda_backend: whether to use the customized cuda backend. By default False, use pure torch version.

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
        self.embeddings, self.n_total_embed, self.offsets, self.resolutions = self.init_embeddings()

        # set volume with base res
        self.volume = Volume(
            n_grid=self.base_res, origin=origin, side=side, xlen=xlen, ylen=ylen, zlen=zlen, dtype=dtype
        )

        # set up cuda backend
        self.use_cuda_backend = use_cuda_backend
        if self.use_cuda_backend and CUDA_BACKEND_AVAILABLE:
            self.hashgrid_encode = HashGridEncode()

        self.out_dim = n_levels * n_feat_per_entry + include_input * input_dim  # L * F + 3

    def get_output_dim(self):
        """Get output dim"""
        return self.out_dim

    def init_embeddings(self, std=1e-4):
        """Init embedding. To save memory, at lower level, do not init large embeddings

        Args:
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
        embeddings = nn.Parameter(torch.empty(n_total_embed, self.n_feat_per_entry))
        nn.init.uniform_(embeddings, -std, std)

        return embeddings, n_total_embed, offsets, resolutions

    def forward(self, xyz: torch.Tensor):
        """
        Args:
            xyz: tensor of shape (B, 3), xyz position

        Returns:
            out: tensor of shape (B, out_dim=T*F + include_inputs * input_dim)
        """
        assert len(xyz.shape) == 2 and xyz.shape[-1] == 3, 'Must be (B, 3) tensor'

        out = []
        if self.include_input:
            out.append(xyz)  # (B, 3)

        if self.use_cuda_backend and CUDA_BACKEND_AVAILABLE and torch.cuda.is_available():
            hashgrid_embed = self.hashgrid_encode(xyz, self.embeddings)
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
            voxel_idx, valid_idx = self.volume.get_voxel_idx_from_xyz(xyz)  # (B, 3), xyz index of volume
            assert voxel_idx.max() < n_grid, 'Voxel idx exceed boundary...'

            if torch.any(valid_idx):
                # get valid grid pts position
                grid_pts_idx_valid = self.volume.get_grid_pts_idx_by_voxel_idx(
                    voxel_idx[valid_idx], flatten=False
                )  # (B_valid, 8, 3)
                grid_pts_valid = self.volume.get_grid_pts_by_voxel_idx(voxel_idx[valid_idx])  # (B_valid, 8, 3)

                # calculate weights to 8 grid_pts by inverse distance
                grid_pts_weights_valid = self.volume.cal_weights_to_grid_pts(
                    xyz[valid_idx], grid_pts_valid
                )  # (B_valid, 8)

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
            idx: (..., 3) index of xyz
            hashmap_size: size of the current hashmap

        Return:
            hash_index: (..., ) hash index
        """
        primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]  # at most 7

        hash_index = torch.zeros_like(idx[..., 0])  # (B,)
        for i in range(idx.shape[-1]):
            hash_index ^= idx[..., i] * primes[i]

        return hash_index % hashmap_size
