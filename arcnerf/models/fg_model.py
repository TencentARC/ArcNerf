# -*- coding: utf-8 -*-

import torch

from .base_3d_model import Base3dModel
from arcnerf.geometry.sphere import Sphere
from arcnerf.geometry.volume import Volume
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class FgModel(Base3dModel):
    """Class for fg model. Child class of Base3dModel
     You can use contain a explicit structure for fast/accurate inner object sampling and ray marching.
     The structure can be a dense volume, octree, sphere, or other structure etc.

     But by default it do not use such bounding structure, and the sampling is in a larger area.

     Any other modeling methods(NeRF, NeuS, mip-nerf) inherit this model and have their detailed sampling/rendering
     algorithms.
    """

    def __init__(self, cfgs):
        super(FgModel, self).__init__(cfgs)
        # optimize the struct(pruning the volume, etc) periodically
        self.epoch_optim = get_value_from_cfgs_field(self.cfgs.model, 'epoch_optim', None)
        self.epoch_optim_warmup = get_value_from_cfgs_field(self.cfgs.model, 'epoch_optim_warmup', None)
        self.ema_optim_decay = get_value_from_cfgs_field(self.cfgs.model, 'ema_optim_decay', 0.95)
        self.opa_thres = get_value_from_cfgs_field(self.cfgs.model, 'opa_thres', 0.01)
        # whether use accelerated sampling or uniform sample in (near, far)
        self.ray_sample_acc = get_value_from_cfgs_field(self.cfgs.model, 'ray_sample_acc', False)
        # inner object bounding structure
        self.obj_bound, self.obj_bound_type = get_value_from_cfgs_field(self.cfgs.model, 'obj_bound'), None
        if self.obj_bound is not None:
            self.set_up_obj_bound_structure()
        # bkg color/depth/normal for invalid rays
        self.bkg_color = get_value_from_cfgs_field(self.cfgs.model, 'bkg_color', [1.0, 1.0, 1.0])  # white
        self.depth_far = get_value_from_cfgs_field(self.cfgs.model, 'depth_far', 10.0)  # far distance
        self.normal = get_value_from_cfgs_field(self.cfgs.model, 'normal', [1.0, 0.0, 0.0])  # for eikonal loss cal

    def get_obj_bound_and_type(self):
        """Get the obj bound and type"""
        return self.obj_bound, self.obj_bound_type

    def set_up_obj_bound_structure(self):
        """Set up the bounding structure of the model"""
        if 'volume' in self.obj_bound.__dict__.keys():
            volume_cfgs = get_value_from_cfgs_field(self.obj_bound, 'volume', None)
            if volume_cfgs is not None:
                self.set_up_volume(volume_cfgs)
        elif 'sphere' in self.obj_bound.__dict__.keys():
            sphere_cfgs = get_value_from_cfgs_field(self.obj_bound, 'sphere', None)
            if sphere_cfgs is not None:
                self.obj_bound_type = 'sphere'
                self.obj_bound = Sphere(**sphere_cfgs.__dict__)
        else:
            raise NotImplementedError('Obj bound type {} not support...'.format(self.obj_bound.__dict__.keys()))

    def set_up_volume(self, volume_cfgs):
        """Set up the dense volume with bitfield"""
        self.obj_bound_type = 'volume'
        if get_value_from_cfgs_field(volume_cfgs, 'n_grid') is None:  # Must set a default resolution
            volume_cfgs.n_grid = 128
        self.obj_bound = Volume(**volume_cfgs.__dict__)
        if self.epoch_optim is not None:  # setup bitfield for pruning
            self.obj_bound.set_up_voxel_bitfield()
            self.obj_bound.set_up_voxel_opafield()

    def get_near_far_from_rays(self, inputs):
        """Get the near/far zvals from rays given settings. If the inner structure for model exists,
        it will sample only in the object area. Otherwise call Base3dModel's method to sample in large area.

        Args:
            inputs: a dict of torch tensor:
                rays_o: torch.tensor (B, 3), cam_loc/ray_start position
                rays_d: torch.tensor (B, 3), view dir(assume normed)
                bounds: torch.tensor (B, 2). optional
        Returns:
            near, far:  torch.tensor (B, 1) each
        """
        if self.obj_bound_type is not None:
            if self.obj_bound_type == 'volume':
                near, far, _, _ = self.obj_bound.ray_volume_intersection(inputs['rays_o'], inputs['rays_d'])
            elif self.obj_bound_type == 'sphere':
                near, far, _, _ = self.obj_bound.ray_sphere_intersection(inputs['rays_o'], inputs['rays_d'])
            else:
                raise NotImplementedError('Ray-{} is not valid...Please implement it...'.format(self.obj_bound_type))
        else:  # call the parent class
            near, far = super().get_near_far_from_rays(inputs)

        return near, far

    def get_zvals_from_near_far(self, near: torch.Tensor, far: torch.Tensor, n_pts, inference_only=False):
        """Get the zvals of the object with/without bounding structure"""
        zvals = None
        if self.obj_bound_type is not None:
            if self.obj_bound_type == 'volume' and self.epoch_optim is not None and self.ray_sample_acc:
                self.get_zvals_from_sparse_volume(near, far, n_pts, inference_only)
                # TODO: Sample in the occupied cells

        # only volume based method with pruning allowed acceleration
        if zvals is None:
            zvals = super().get_zvals_from_near_far(near, far, n_pts, inference_only)

        return zvals

    def get_zvals_from_sparse_volume(self, near: torch.Tensor, far: torch.Tensor, n_pts, inference_only=False):
        """Get the zvals from optimized coarse volume"""
        pass  # TODO

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        """If you use a geometric structure bounding the object, some rays does not hit the bound can be ignored.
         You can assign a bkg color to them directly, with opacity=0.0 and depth=some far distance.
        """
        # optimize the obj bound, mostly for volume base structure
        if cur_epoch > 0 and self.epoch_optim is not None and cur_epoch % self.epoch_optim == 0:
            self.optim_obj_bound(cur_epoch)

        # TODO: PUT this as a super call in subclass

    @torch.no_grad()
    def optim_obj_bound(self, cur_epoch):
        """Optimize the obj bounding geometric structure. Support ['volume'] now."""
        if self.obj_bound_type == 'volume':
            self.optim_volume(cur_epoch)

    def optim_volume(self, cur_epoch):
        """Optimize the dense volume by sampling points in the voxel.
         In warmup stage, sample all cells and update
         Else in pose-warmup stage, uniform sampled 1/4 cells from all and 1/4 cells from occupied cells.
         """
        volume = self.obj_bound
        if self.epoch_optim_warmup is not None and cur_epoch < self.epoch_optim_warmup:
            voxel_idx = volume.get_full_voxel_idx(flatten=True)  # (N_grid**3, 3)
            voxel_pts = volume.get_volume_pts()  # (N_grid**3, 3)
        else:
            n_grid = volume.get_n_grid()
            n_sample = volume.get_n_voxel() // 4  # (N_grid**3) / 4
            # 1/4 uniform from (0~n_grid)^3
            uni_voxel_idx = torch.randperm(
                volume.get_n_voxel(), dtype=torch.long, device=volume.get_device()
            )[:n_sample]  # (N_sample,)
            uni_voxel_idx = volume.convert_flatten_index_to_xyz_index(uni_voxel_idx, n_grid)  # (N_sample, 3)
            # 1/4 occupied cells
            occ_voxel_idx = volume.get_occupied_voxel_idx()[:n_sample, :]  # (N_sample, 3)

            voxel_idx = torch.cat([uni_voxel_idx, occ_voxel_idx], dim=0)  # (2*N_sample, 3)
            voxel_pts = volume.get_voxel_pts_by_voxel_idx(voxel_idx)  # (2*N_sample, 3)

        # add noise to perturb in the voxel
        dtype = voxel_pts.dtype
        device = voxel_pts.device
        noise = torch.rand_like(voxel_pts, dtype=dtype, device=device) - 0.5  # (N, 3) in (-1/2, 1/2)
        noise *= (volume.get_voxel_size(to_list=False)[None, :])  # (N, 3) in (-v_s/2, v_s/2)
        voxel_pts += noise

        # get the opacity
        dt = volume.get_diag_len() / float(self.get_ray_cfgs('n_sample'))  # only consider n_sample pts
        opacity = self.get_est_opacity(dt, voxel_pts)  # (N,)

        # update opacity and bitfield
        volume.update_opafield_by_voxel_idx(voxel_idx, opacity, ema=self.ema_optim_decay)
        volume.update_bitfield_by_opafield(threshold=self.opa_thres)

    def get_est_opacity(self, dt, pts):
        """Get the estimated opacity at certain pts. This method is only for fg_model.
        In density model, when density is high, opacity = 1 - exp(-sigma*dt), when sigma is large, opacity is large.
        You have to rewrite this function in sdf-like models

        Args:
            dt: the dt used for calculated
            pts: the pts in the field. (B, 3) xyz position. Need geometric model to process

        Returns:
            opacity: (B,) opacity. In density model, opacity = 1 - exp(-sigma*dt)
                                   For sdf model,  opacity = 1 - exp(-sdf_to_sigma(sdf)*dt)
            When opacity is large(Than some thresold), pts can be considered as in the object.
        """
        density = self.forward_pts(pts)  # (B,)
        opacity = 1.0 - torch.exp(-torch.relu(density) * dt)  # (B,)

        return opacity
