# -*- coding: utf-8 -*-

import torch

from .base_3d_model import Base3dModel
from arcnerf.geometry.ray import surface_ray_intersection
from arcnerf.geometry.sphere import Sphere
from arcnerf.geometry.transformation import normalize
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
     algorithms. This model is used to provide the near-accurate sampling in constrained space.
    """

    def __init__(self, cfgs):
        super(FgModel, self).__init__(cfgs)
        # inner object bounding structure
        self.optim_cfgs = self.read_optim_cfgs()
        self.obj_bound, self.obj_bound_type = get_value_from_cfgs_field(self.cfgs.model, 'obj_bound'), None
        if self.obj_bound is not None:
            self.set_up_obj_bound_structure()

    def read_optim_cfgs(self):
        """Read optim params under model.obj_bound. Prams controls optimization"""
        params = {}
        if get_value_from_cfgs_field(self.cfgs.model, 'obj_bound') is None:
            params['epoch_optim'] = None
            params['ray_sample_acc'] = False
            # default values
            params['bkg_color'] = [1.0, 1.0, 1.0]  # white
            params['depth_far'] = 10.0  # far distance
            params['normal'] = [0.0, 1.0, 0.0]  # for eikonal loss cal, up direction
        else:
            cfgs = self.cfgs.model.obj_bound
            params['epoch_optim'] = get_value_from_cfgs_field(cfgs, 'epoch_optim', None)
            params['epoch_optim_warmup'] = get_value_from_cfgs_field(cfgs, 'epoch_optim_warmup', None)
            params['ema_optim_decay'] = get_value_from_cfgs_field(cfgs, 'ema_optim_decay', 0.95)
            params['opa_thres'] = get_value_from_cfgs_field(cfgs, 'opa_thres', 0.01)
            # whether use accelerated sampling or uniform sample in (near, far)
            params['ray_sample_acc'] = get_value_from_cfgs_field(cfgs, 'ray_sample_acc', False)
            # bkg color/depth/normal for invalid rays
            params['bkg_color'] = get_value_from_cfgs_field(self.cfgs, 'bkg_color', [1.0, 1.0, 1.0])  # white
            params['depth_far'] = get_value_from_cfgs_field(self.cfgs, 'depth_far', 10.0)  # far distance
            params['normal'] = get_value_from_cfgs_field(self.cfgs, 'normal', [0.0, 1.0, 0.0])  # for eikonal loss cal

        return params

    def get_optim_cfgs(self, key=None):
        """Get optim cfgs by optional key"""
        if key is None:
            return self.optim_cfgs

        return self.optim_cfgs[key]

    def set_optim_cfgs(self, key, value):
        """Set optim cfgs by key"""
        self.optim_cfgs[key] = value

    def get_n_coarse_sample(self):
        """Num of coarse sample for sampling in the foreground space. By default use n_sample in configs"""
        return self.get_ray_cfgs('n_sample')

    def get_obj_bound_and_type(self):
        """Get the obj bound and type"""
        return self.obj_bound, self.obj_bound_type

    def set_up_obj_bound_structure(self):
        """Set up the bounding structure of the model"""
        self.set_up_obj_bound_structure_by_cfgs(self.obj_bound)

    def set_up_obj_bound_structure_by_cfgs(self, cfgs):
        """Set up the bounding structure by cfgs. Can call it outside"""
        if 'volume' in cfgs.__dict__.keys():
            volume_cfgs = get_value_from_cfgs_field(cfgs, 'volume', None)
            if volume_cfgs is not None:
                self.set_up_volume(volume_cfgs)
        elif 'sphere' in cfgs.__dict__.keys():
            sphere_cfgs = get_value_from_cfgs_field(cfgs, 'sphere', None)
            if sphere_cfgs is not None:
                self.obj_bound_type = 'sphere'
                self.obj_bound = Sphere(**sphere_cfgs.__dict__)
        else:
            raise NotImplementedError('Obj bound type {} not support...'.format(cfgs.__dict__.keys()))

    def set_up_volume(self, volume_cfgs):
        """Set up the dense volume with bitfield"""
        self.obj_bound_type = 'volume'
        if get_value_from_cfgs_field(volume_cfgs, 'n_grid') is None:  # Must set a default resolution
            volume_cfgs.n_grid = 128
        self.obj_bound = Volume(**volume_cfgs.__dict__)
        if self.get_optim_cfgs('epoch_optim') is not None:  # setup bitfield for pruning
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
            mask: (B,) of each rays information
        """
        if self.obj_bound_type is not None:
            if self.obj_bound_type == 'volume':
                in_occ = self.get_optim_cfgs('epoch_optim') is not None
                near, far, _, mask = self.obj_bound.ray_volume_intersection(inputs['rays_o'], inputs['rays_d'], in_occ)
            elif self.obj_bound_type == 'sphere':
                near, far, _, mask = self.obj_bound.ray_sphere_intersection(inputs['rays_o'], inputs['rays_d'])
            else:
                raise NotImplementedError('Ray-{} is not valid...Please implement it...'.format(self.obj_bound_type))
        else:  # call the parent class
            near, far, mask = super().get_near_far_from_rays(inputs)

        mask = None if mask is None else mask[:, 0]

        return near, far, mask

    def get_zvals_from_near_far(self, near: torch.Tensor, far: torch.Tensor, n_pts, inference_only=False):
        """Get the zvals of the object with/without bounding structure
            If ray_sample_acc, Skip empty voxels to sample max up to n_pts points by some step.
            Else, find the rays's intersection with remaining voxels, and use near, far to sampling directly
        """
        zvals = None
        if self.obj_bound_type is not None:
            if self.obj_bound_type == 'volume' and self.get_optim_cfgs('epoch_optim') is not None\
                    and self.get_optim_cfgs('ray_sample_acc'):
                self.get_zvals_from_sparse_volume(near, far, n_pts, inference_only)

        # only volume based method with pruning allowed acceleration
        if zvals is None:
            zvals = super().get_zvals_from_near_far(near, far, n_pts, inference_only)

        return zvals

    def get_zvals_from_sparse_volume(self, near: torch.Tensor, far: torch.Tensor, n_pts, inference_only=False):
        """Get the zvals from optimized coarse volume which skip the empty voxels """
        pass

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        """If you use a geometric structure bounding the object, some rays does not hit the bound can be ignored.
         You can assign a bkg color to them directly, with opacity=0.0 and depth=some far distance.
         TODO: We can only support fix num of sampling for each rays now. Is this good enough?
        """
        # find the near/far and mask
        near, far, mask = self.get_near_far_from_rays(inputs)

        if self.obj_bound_type is None or torch.all(mask):  # all the rays to run
            zvals = self.get_zvals_from_near_far(near, far, self.get_n_coarse_sample(), inference_only)
            output = self._forward(inputs, zvals, inference_only, get_progress, cur_epoch, total_epoch)
        else:  # only on valid rays
            zvals_valid = self.get_zvals_from_near_far(
                near[mask], far[mask], self.get_n_coarse_sample(), inference_only
            )

            inputs_valid = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs_valid[k] = v[mask]
                else:
                    inputs_valid[k] = v

            output_valid = self._forward(
                inputs_valid, zvals_valid, inference_only, get_progress, cur_epoch, total_epoch
            )

            # update invalid rays by default values
            output = self.update_default_values_for_invalid_rays(output_valid, mask)

        return output

    def update_default_values_for_invalid_rays(self, output_valid, mask: torch.Tensor):
        """Update the default values for invalid rays

        Args:
            output_valid: that contains keys with `rgb*`/`mask*`/`depth*`/`normal*`/`progress_*` in (N_valid, ...) shape
            For each keys, fill with
                rgb/rgb_*: by self.optim_params['bkg_color'], by default (1, 1, 1) as white bkg
            mask: mask indicating each rays' validity. in (N_rays, ), with B_valid `True` values.

        Returns:
            output: same keys with output_valid with filled in values. Each tensor will be (N_rays, ...)
        """
        n_rays = mask.shape[0]
        output = {}

        for k, v in output_valid.items():
            if not isinstance(v, torch.Tensor):
                output[k] = v
            else:
                dtype = v.dtype
                device = v.device
                new_shape = (n_rays, *v.shape[1:])
                if k.startswith('rgb'):  # update bkg_color
                    bkg_color = self.get_optim_cfgs('bkg_color')
                    bkg_color = torch.tensor(bkg_color, dtype=dtype, device=device)[None]  # (1, 3)
                    out_tensor = torch.ones(new_shape, dtype=dtype, device=device) * bkg_color
                    out_tensor[mask] = v
                    output[k] = out_tensor
                elif k.startswith('depth'):  # update depth
                    depth_far = self.get_optim_cfgs('depth_far')
                    depth_far = torch.tensor(depth_far, dtype=dtype, device=device)[None]  # (1, 1)
                    out_tensor = torch.ones(new_shape, dtype=dtype, device=device) * depth_far
                    out_tensor[mask] = v
                    output[k] = out_tensor
                elif k.startswith('mask'):  # update mask, must be 0
                    out_tensor = torch.zeros(new_shape, dtype=dtype, device=device)
                    out_tensor[mask] = v
                    output[k] = out_tensor
                elif k.startswith('normal'):  # update normal
                    normal = self.get_optim_cfgs('normal')
                    normal = torch.tensor(normal, dtype=dtype, device=device)[None]  # (1, 3)
                    normal = normalize(normal)
                    if k == 'normal_pts':
                        normal = normal[None]  # (1, 3, 3)
                    out_tensor = torch.ones(new_shape, dtype=dtype, device=device) * normal
                    out_tensor[mask] = v
                    output[k] = out_tensor
                elif k.startswith('progress'):  # in (N_valid, N_pts, ...)
                    if 'sigma' in k and self.sigma_reverse():  # `sdf`
                        out_tensor = -torch.ones(new_shape, dtype=dtype, device=device)  # make it outside
                        out_tensor[mask] = v
                        output[k] = out_tensor
                    else:  # for all `sigma`/`zvals`/`alpha`/`trans_shift`/`weights`/`radiance`, they are all zeros
                        out_tensor = torch.zeros(new_shape, dtype=dtype, device=device)
                        out_tensor[mask] = v
                        output[k] = out_tensor
                else:
                    output[k] = v

        return output

    def _forward(self, inputs, zvals, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        """The method that really process all rays that have intersection with the bound

        Args:
            zvals: it is the valid coarse zvals get from foreground model.
                    If no obj_bound is provided, it uses near/far and bounding_radius to calculate in a large space
                    If obj_bound is volume/sphere, it use the zvals that rays hits the structure.
        """
        raise NotImplementedError(
            'You should implement the _forward function that process rays with coarse zvals in child class...'
        )

    @torch.no_grad()
    def optimize(self, cur_epoch=0):
        """Optimize the obj bounding geometric structure. Support ['volume'] now."""
        epoch_optim = self.get_optim_cfgs('epoch_optim')
        if cur_epoch > 0 and epoch_optim is not None and cur_epoch % epoch_optim == 0:
            if self.obj_bound_type == 'volume':
                self.optim_volume(cur_epoch)

    def optim_volume(self, cur_epoch):
        """Optimize the dense volume by sampling points in the voxel.
         In warmup stage, sample all cells and update
         Else in pose-warmup stage, uniform sampled 1/4 cells from all and 1/4 cells from occupied cells.
         """
        volume = self.obj_bound
        epoch_optim_warmup = self.get_optim_cfgs('epoch_optim_warmup')
        if epoch_optim_warmup is not None and cur_epoch < epoch_optim_warmup:
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
        volume.update_opafield_by_voxel_idx(voxel_idx, opacity, ema=self.get_optim_cfgs('ema_optim_decay'))
        volume.update_bitfield_by_opafield(threshold=self.get_optim_cfgs('opa_thres'))

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

    def surface_render(
        self, inputs, method='sphere_tracing', n_step=128, n_iter=100, threshold=0.01, level=50.0, grad_dir='descent'
    ):
        """Surface rendering by finding the surface point and its color. Only for inference
        Compare to the func in parent class(base_3d_model, you need to consider the case that some rays do not hit the
        obj_bound structure, with helps to save computation.
        """
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        dtype = rays_o.dtype
        device = rays_o.device
        n_rays = rays_o.shape[0]

        # get bounds for object
        near, far, valid_rays = self.get_near_far_from_rays(inputs)  # (B, 1) * 2

        # get the network
        geo_net, radiance_net = self.get_net()

        # get surface pts for valid_rays
        if valid_rays is None or torch.all(valid_rays):
            zvals, pts, mask = surface_ray_intersection(
                rays_o, rays_d, geo_net.forward_geo_value, method, near, far, n_step, n_iter, threshold, level, grad_dir
            )
        else:
            zvals_valid, pts_valid, mask_valid = surface_ray_intersection(
                rays_o[valid_rays], rays_d[valid_rays], geo_net.forward_geo_value, method, near[valid_rays],
                far[valid_rays], n_step, n_iter, threshold, level, grad_dir
            )

            # full output update by valid rays
            zvals = torch.ones((n_rays, 1), dtype=dtype, device=device) * zvals_valid.max()
            pts = torch.ones((n_rays, 3), dtype=dtype, device=device)
            mask = torch.zeros((n_rays, ), dtype=torch.bool, device=device)
            zvals[valid_rays], pts[valid_rays], mask[valid_rays] = zvals_valid, pts_valid, mask_valid

        rgb = torch.ones((n_rays, 3), dtype=dtype, device=device)  # white bkg
        depth = zvals  # at max zvals after far
        mask_float = mask.type(dtype)

        # in case all rays do not hit the surface
        if torch.any(mask):
            # forward mask pts/dir for color
            _, rgb_mask = self._forward_pts_dir(geo_net, radiance_net, pts[mask], rays_d[mask])
            rgb[mask] = rgb_mask

        output = {
            'rgb': rgb,  # (B, 3)
            'depth': depth,  # (B,)
            'mask': mask_float,  # (B,)
        }

        return output
