# -*- coding: utf-8 -*-

import torch

from . import BOUND_REGISTRY
from .basic_bound import BasicBound
from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.volume import Volume
from arcnerf.ops.volume_func import sparse_volume_sampling, tensor_reduce_max, CUDA_BACKEND_AVAILABLE
from arcnerf.render.ray_helper import handle_valid_mask_zvals, get_zvals_from_near_far_fix_step
from common.utils.cfgs_utils import valid_key_in_cfgs, get_value_from_cfgs_field


@BOUND_REGISTRY.register()
class VolumeBound(BasicBound):
    """A volume structure bounding the object"""

    def __init__(self, cfgs):
        super(VolumeBound, self).__init__(cfgs)
        assert valid_key_in_cfgs(cfgs, 'volume'), 'You must have volume in the cfgs'

        self.cfgs = cfgs
        self.read_optim_cfgs()

        # set up the volume
        volume_cfgs = cfgs.volume
        if get_value_from_cfgs_field(volume_cfgs, 'n_grid') is None:
            cfgs.n_grid = 128
        self.volume = Volume(**volume_cfgs.__dict__)

        # set bitfield for pruning
        if self.get_optim_cfgs('epoch_optim') is not None:  # setup bitfield for pruning
            self.volume.set_up_voxel_bitfield(init_occ=True)
            self.volume.set_up_voxel_opafield()

    def get_obj_bound(self):
        """Get the real obj bounding structure"""
        return self.volume

    def read_optim_cfgs(self):
        """Read optim params under model.obj_bound. Prams controls optimization"""
        params = super().read_optim_cfgs()

        # whether use accelerated sampling or uniform sample in (near, far)
        params['ray_sample_acc'] = get_value_from_cfgs_field(self.cfgs, 'ray_sample_acc', False)
        # whether to use fix step for zvals
        params['ray_sample_fix_step'] = get_value_from_cfgs_field(self.cfgs, 'ray_sample_fix_step', False)
        params['near_distance'] = get_value_from_cfgs_field(self.cfgs, 'near_distance', 0.0)

        return params

    def get_near_far_from_rays(self, inputs, **kwargs):
        """Get the near/far zvals from rays using volume bounding. Coarsely sample is allow

        Returns:
            near, far: torch.tensor (B, 1) each
            mask_rays: torch.tensor (B,), each rays validity
        """
        # in the coarse volume to save aabb cal time
        near, far, _, mask_rays = self.volume.ray_volume_intersection(inputs['rays_o'], inputs['rays_d'])

        return near, far, mask_rays[:, 0]

    def get_zvals_from_near_far(
        self,
        near,
        far,
        n_pts,
        inference_only=False,
        inverse_linear=False,
        perturb=False,
        rays_o=None,
        rays_d=None,
        **kwargs
    ):
        """If the volume is being prunned, we can sample coarsely in remaining voxel.

        If ray_sample_acc, Skip empty voxels to sample max up to n_pts points by some step.
        Else, find the rays's intersection with remaining voxels, and use near, far to sampling directly

        rays_o/rays_d is for coarse sampling in the volume

        Returns:
            near, far: torch.tensor (B, 1) each
            mask_pts: torch.tensor (B, n_pts), each pts validity on each ray.
                    This helps the following network reduces duplicated computation.

        """
        if self.get_optim_cfgs('epoch_optim') is not None and self.get_optim_cfgs('ray_sample_acc'):
            return self.get_zvals_from_sparse_volume(
                rays_o, rays_d, near, far, n_pts, inference_only, inverse_linear, perturb
            )
        else:
            return super().get_zvals_from_near_far(near, far, n_pts, inference_only, inverse_linear, perturb)

    @torch.no_grad()
    def get_zvals_from_sparse_volume(self, rays_o, rays_d, near, far, n_pts, inference_only, inverse_linear, perturb):
        """Get the zvals from optimized coarse volume which skip the empty voxels

        The zvals are tensor in (N_rays, N_pts). But coarse sampling makes some rays do not have n_pts max sampling.
        For those rays, it will use the same zvals as the far.
        It the ray does not contains any sampled pts, it will be all 0 in the row.

        Returns:
            zvals: (B, n_pts) tensor of zvals
            mask_pts: (B, n_pts) bool tensor of all the pts
        """
        if CUDA_BACKEND_AVAILABLE:  # use the customized cuda volume sampling method, will be fast
            const_dt = self.volume.get_diag_len() / n_pts
            zvals, mask_pts = sparse_volume_sampling(
                rays_o,
                rays_d,
                near,
                far,
                n_pts,
                const_dt,
                self.volume.get_range(),
                self.volume.get_n_grid(),
                self.volume.get_voxel_bitfield(),
                near_distance=self.get_optim_cfgs('near_distance')
            )
        else:  # easy sampling in pure torch
            if self.get_optim_cfgs('ray_sample_fix_step'):  # fix step sampling
                zvals, mask_pts = self.get_zvals_from_near_far_fix_step(
                    near, far, n_pts, inference_only, perturb
                )  # (N_rays, N_pts)
                pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (N_rays, N_pts, 3)
                pts_valid = pts[mask_pts].view(-1, 3)  # (N_valid_pts, 3)
                mask_valid_pts = self.volume.check_pts_in_occ_voxel(pts_valid)  # (N_valid_pts, 3)
                # update those pts not in occ voxel
                mask_pts[mask_pts.clone()] = torch.logical_and(mask_pts[mask_pts.clone()], mask_valid_pts)
            else:  # near/far uniform sampling and mask not in bound pts
                zvals, _ = super().get_zvals_from_near_far(
                    near, far, n_pts, inference_only, inverse_linear, perturb
                )  # (N_rays, N_pts)
                pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (N_rays, N_pts, 3)
                pts = pts.view(-1, 3)  # (N_rays*N_pts, 3)
                mask_pts = self.volume.check_pts_in_occ_voxel(pts)  # (N_rays*N_pts,)
                mask_pts = mask_pts.view(-1, n_pts)  # (N_rays, N_pts)

            # realign the valid zvals and mask_pts
            zvals, mask_pts = handle_valid_mask_zvals(zvals, mask_pts)

        return zvals, mask_pts

    def get_zvals_from_near_far_fix_step(self, near, far, n_pts, inference_only=False, perturb=False, **kwargs):
        """Get zvals from near to far with fix step. Use this will sample less pts
        compared to directly sample in (near, far) uniformly

        Returns:
            zvals: (B, n_pts) tensor of zvals
            mask_pts: (B, n_pts) bool tensor of all the pts
        """
        fix_t = self.volume.get_diag_len() / n_pts  # diag len based
        zvals, mask_pts = get_zvals_from_near_far_fix_step(
            near, far, fix_t, n_pts, perturb=perturb if not inference_only else False
        )

        return zvals, mask_pts

    @torch.no_grad()
    def optimize(self, cur_epoch=0, n_pts=128, get_est_opacity=None):
        """Overwrite the optimization function for volume pruning

        In warmup stage, sample all cells and update
        Else in pose-warmup stage, uniform sampled 1/4 cells from all and 1/4 cells from occupied cells.
        """
        epoch_optim = self.get_optim_cfgs('epoch_optim')
        epoch_optim_warmup = self.get_optim_cfgs('epoch_optim_warmup')

        if cur_epoch <= 0 or epoch_optim is None or cur_epoch % epoch_optim != 0:
            return

        # select pts randomly in some voxels
        if epoch_optim_warmup is not None and cur_epoch < epoch_optim_warmup:
            voxel_idx = self.volume.get_full_voxel_idx(flatten=True)  # (N_grid**3, 3)
            voxel_pts = self.volume.get_volume_pts().clone()  # (N_grid**3, 3)
        else:
            n_grid = self.volume.get_n_grid()
            n_sample = self.volume.get_n_voxel() // 4  # (N_grid**3) / 4
            # 1/4 uniform from (0~n_grid)^3
            uni_voxel_idx = torch.randperm(
                self.volume.get_n_voxel(), dtype=torch.long, device=self.volume.get_device()
            )[:n_sample]  # (N_sample,)
            uni_voxel_idx = self.volume.convert_flatten_index_to_xyz_index(uni_voxel_idx, n_grid)  # (N_sample, 3)
            # 1/4 occupied cells
            occ_voxel_idx = self.volume.get_occupied_voxel_idx()[:n_sample, :]  # (N_sample, 3)

            voxel_idx = torch.cat([uni_voxel_idx, occ_voxel_idx], dim=0)  # (2*N_sample, 3)
            voxel_pts = self.volume.get_voxel_pts_by_voxel_idx(voxel_idx).clone()  # (2*N_sample, 3)

        # add noise to perturb in the voxel
        dtype = voxel_pts.dtype
        device = voxel_pts.device
        noise = torch.rand_like(voxel_pts, dtype=dtype, device=device) - 0.5  # (N, 3) in (-1/2, 1/2)
        noise *= (self.volume.get_voxel_size(to_list=False)[None, :])  # (N, 3) in (-v_s/2, v_s/2)
        voxel_pts += noise

        # get the opacity by call the funcition
        dt = self.volume.get_diag_len() / float(n_pts)  # only consider n coarse sample pts
        opacity = get_est_opacity(dt, voxel_pts)  # (N,)

        # max opa in the same group, official used index_reduce_('amax') but this is not support in lower torch version
        uni_voxel_idx, idx = torch.unique(voxel_idx, dim=0, return_inverse=True)
        uni_opa = torch.zeros((uni_voxel_idx.shape[0], ), dtype=dtype, device=device).index_add_(0, idx, opacity)

        if CUDA_BACKEND_AVAILABLE:  # index_reduce_('amax')
            # handle by max
            uni_opa = tensor_reduce_max(opacity, idx, uni_voxel_idx.shape[0])

        # update opacity and bitfield
        self.volume.update_opafield_by_voxel_idx(uni_voxel_idx, uni_opa, ema=self.get_optim_cfgs('ema_optim_decay'))
        self.volume.update_bitfield_by_opafield(threshold=self.get_optim_cfgs('opa_thres'), ops='overwrite')
