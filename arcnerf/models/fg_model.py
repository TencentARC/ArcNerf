# -*- coding: utf-8 -*-

import torch

from .base_3d_model import Base3dModel
from .base_modules.obj_bound import build_obj_bound
from arcnerf.geometry.ray import surface_ray_intersection, get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing


@MODEL_REGISTRY.register()
class FgModel(Base3dModel):
    """Class for fg model. Child class of Base3dModel
     You can use contain a explicit structure for fast/accurate inner object sampling and ray marching.
     The structure can be a dense volume, octree, sphere, or other structure etc.

     But by default it do not use such bounding structure, and the sampling is in a larger area.

     Any other modeling methods(NeRF, NeuS, mip-nerf) inherit this model and have their detailed sampling/rendering
     algorithms. This model is used to provide the near-accurate sampling in constrained space.

     This class focus on the logic of handling rays that not hit on the inner structure, and fill default valid with
     the invalid rays. The detail of sampling is in each Bound class.
    """

    def __init__(self, cfgs):
        super(FgModel, self).__init__(cfgs)
        # inner object bounding structure
        self.obj_bound, self.obj_bound_type = build_obj_bound(cfgs.model)
        self.render_cfgs = self.read_render_cfgs()

    def read_render_cfgs(self):
        """Read render params under model.obj_bound. Params used to fill invalid rays"""
        params = {}
        if get_value_from_cfgs_field(self.cfgs.model, 'obj_bound') is None:
            # default values
            params['bkg_color'] = [1.0, 1.0, 1.0]  # white
            params['depth_far'] = 10.0  # far distance
            params['normal'] = [0.0, 1.0, 0.0]  # for eikonal loss cal, up direction
        else:
            cfgs = self.cfgs.model.obj_bound
            # bkg color/depth/normal for invalid rays
            params['bkg_color'] = get_value_from_cfgs_field(cfgs, 'bkg_color', [1.0, 1.0, 1.0])  # white
            params['depth_far'] = get_value_from_cfgs_field(cfgs, 'depth_far', 10.0)  # far distance
            params['normal'] = get_value_from_cfgs_field(cfgs, 'normal', [0.0, 1.0, 0.0])  # for eikonal loss cal

        return params

    def get_render_cfgs(self, key=None):
        """Get render cfgs by optional key"""
        if key is None:
            return self.render_cfgs

        return self.render_cfgs[key]

    def set_render_cfgs(self, key, value):
        """Set render cfgs by key"""
        self.render_cfgs[key] = value

    def set_up_obj_bound_by_cfgs(self, cfgs):
        """Manually setup the obj bound."""
        self.obj_bound, self.obj_bound_type = build_obj_bound(cfgs)

    def get_n_coarse_sample(self):
        """Num of coarse sample for sampling in the foreground space. By default use n_sample in configs"""
        return self.get_ray_cfgs('n_sample')

    def get_obj_bound(self):
        """Get the real obj bound """
        return self.obj_bound

    def get_obj_bound_type(self):
        """Get the obj bound type"""
        return self.obj_bound_type

    def get_obj_bound_structure(self):
        """Get the real obj bound structure """
        return self.obj_bound.get_obj_bound()

    def get_optim_cfgs(self, key=None):
        """Get optim cfgs by optional key in the obj bound class"""
        return self.obj_bound.get_optim_cfgs(key)

    def set_optim_cfgs(self, key, value):
        """Set optim cfgs by optional key in the obj bound class"""
        return self.obj_bound.set_optim_cfgs(key, value)

    @torch.no_grad()
    def get_near_far_from_rays(self, inputs):
        """Call the get near_far func in bounding class. Allow them to access the ray cfgs"""
        return self.obj_bound.get_near_far_from_rays(
            inputs,
            near_hardcode=self.get_ray_cfgs('near'),
            far_hardcode=self.get_ray_cfgs('far'),
            bounding_radius=self.get_ray_cfgs('bounding_radius')
        )

    @torch.no_grad()
    def get_zvals_from_near_far(self, near, far, n_pts, inference_only=False, rays_o=None, rays_d=None):
        """Call the get zvals func in bounding class. Allow them to access the ray cfgs"""
        return self.obj_bound.get_zvals_from_near_far(
            near,
            far,
            n_pts,
            inference_only,
            self.get_ray_cfgs('inverse_linear'),
            self.get_ray_cfgs('perturb'),
            rays_o=rays_o,
            rays_d=rays_d
        )

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        """If you use a geometric structure bounding the object, some rays does not hit the bound can be ignored.
         You can assign a bkg color to them directly, with opacity=0.0 and depth=some far distance.
         This function handles the invalid rays.

         Uf you are using a volume with optimization, we need to mask the rays coarsely using dense volume,
         then find the pts on each rays in a refined manner using pruning volume.
         They skip those rays with empty pts for processing.
        """
        output = {}
        rays_o, rays_d = inputs['rays_o'], inputs['rays_d']
        bkg_color = inputs['bkg_color']

        # find the near/far/zvals and mask of rays and pts
        near, far, mask_rays = self.get_near_far_from_rays(inputs)
        zvals, mask_pts = self.get_zvals_from_near_far(
            near, far, self.get_n_coarse_sample(), inference_only, rays_o, rays_d
        )

        # put them to the input keys
        inputs['zvals'] = zvals
        inputs['mask_pts'] = mask_pts

        # mask_rays: (B, ) / mask_pts: (B, n_pts)
        if mask_rays is None and mask_pts is None:  # process all the rays
            output = self._forward(inputs, inference_only, get_progress, cur_epoch, total_epoch)
        elif mask_rays is None:
            raise RuntimeError('This case should not happen...Check it')
        elif mask_pts is None:
            pass
        else:  # both not None, update real mask_rays
            mask_rays = torch.logical_and(mask_rays, torch.any(mask_pts, dim=1))  # update the mask_rays

        # handle the case of sparse rays
        if mask_rays is not None:
            if torch.all(mask_rays):
                output = self._forward(inputs, inference_only, get_progress, cur_epoch, total_epoch)
            else:
                zvals_valid = zvals[mask_rays]
                mask_pts_valid = mask_pts[mask_rays] if mask_pts is not None else None

                # rare case, the batch send in are all from background
                empty_batch = False
                if torch.sum(mask_rays) == 0:
                    empty_batch = True
                    mask_rays[0] = True  # mask a valid rays to run
                    zvals_valid = torch.zeros((1, zvals.shape[1]), dtype=zvals.dtype, device=zvals.device)
                    zvals_valid[0, 1:] = 1.0
                    if mask_pts is not None:
                        mask_pts_valid = torch.zeros((1, mask_pts.shape[1]), dtype=torch.bool, device=mask_pts.device)
                        mask_pts_valid[0, :2] = True

                inputs_valid = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs_valid[k] = v[mask_rays]
                    else:
                        inputs_valid[k] = v

                # force to use the revised case
                inputs_valid['zvals'] = zvals_valid
                inputs_valid['mask_pts'] = mask_pts_valid
                inputs_valid['bkg_color'] = bkg_color[mask_rays] if bkg_color is not None else None

                output_valid = self._forward(inputs_valid, inference_only, get_progress, cur_epoch, total_epoch)

                if empty_batch:
                    mask_rays[0] = False  # force to synthetic ray to use all default value

                # update invalid rays by default values
                output = self.update_default_values_for_invalid_rays(output_valid, mask_rays, bkg_color)

        return output

    def _forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        """The method that really process all rays that have intersection with the bound

        Args:
            inputs: valid rays with (B, 3) shape and other fields like mask/rgb
                zvals: it is the valid coarse zvals get from foreground model. (B, N_pts) tensor
                        If no obj_bound is provided, it uses near/far and bounding_radius to calculate in a large space
                        If obj_bound is volume/sphere, it use the zvals that rays hits the structure.
                mask_pts: It is a tensor that indicator the validity of pts on each ray. (B, N_pts) tensor
                        False will at the end of each ray indicating they are the same as far pts.
                        If None, all the pts are valid.
                        This helps the child network to process pts without duplication.
                bkg_color: It is a tensor that used to attach bkg_color to the rendering output. (B, 3) tensor
                           If None, do not multiply the color
                           In training, we can use random bkg color to accelerate the converge of synthetic scenes
        """
        raise NotImplementedError(
            'You should implement the _forward function that process rays with coarse zvals in child class...'
        )

    def get_sigma_radiance_by_mask_pts(self, geo_net, radiance_net, rays_o, rays_d, zvals, mask_pts=None):
        """Process the pts/dir by mask_pts. Only process valid zvals to save computation

        Args:
            geo_net: geometry net
            radiance_net: radiance net
            rays_o: (B, 3) rays origin
            rays_d: (B, 3) rays direction(normalized)
            zvals: (B, N_pts) zvals on each ray
            mask_pts: (B, N_pts) whether each pts is valid. If None, process all the pts

        Returns:
            sigma: (B, N_pts) sigma on all pts. Duplicated pts share the same value
            radiance: (B, N_pts, 3) rgb on all pts. Duplicated pts share the same value
        """
        n_rays = zvals.shape[0]
        n_pts = zvals.shape[1]
        dtype = zvals.dtype
        device = zvals.device

        # get points, expand rays_d to all pts
        pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (B, N_pts, 3)
        rays_d_repeat = torch.repeat_interleave(rays_d.unsqueeze(1), n_pts, dim=1)  # (B, N_pts, 3)

        if mask_pts is None:
            pts = pts.view(-1, 3)  # (B*N_pts, 3)
            rays_d_repeat = rays_d_repeat.view(-1, 3)  # (B*N_pts, 3)
        else:
            pts = pts[mask_pts].view(-1, 3)  # (N_valid_pts, 3)
            rays_d_repeat = rays_d_repeat[mask_pts].view(-1, 3)  # (N_valid_pts, 3)

        # get sigma and rgb, . shape in (N_valid_pts, ...)
        _sigma, _radiance = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, geo_net, radiance_net, pts, rays_d_repeat
        )

        # reshape to (B, N_sample, ...) by fill duplicating pts
        if mask_pts is None:
            sigma = _sigma.view(n_rays, -1)  # (B, N_sample)
            radiance = _radiance.view(n_rays, -1, 3)  # (B, N_sample, 3)
        else:
            last_pts_idx = torch.cumsum(mask_pts.sum(dim=1), dim=0) - 1  # index on flatten sigma/radiance
            last_sigma, last_radiance = _sigma[last_pts_idx], _radiance[last_pts_idx]  # (B,) (B, 3)
            sigma = torch.ones((n_rays, n_pts), dtype=dtype, device=device) * last_sigma.unsqueeze(1)
            radiance = torch.ones((n_rays, n_pts, 3), dtype=dtype, device=device) * last_radiance.unsqueeze(1)
            sigma[mask_pts] = _sigma
            radiance[mask_pts] = _radiance

        return sigma, radiance

    def update_default_values_for_invalid_rays(self, output_valid, mask, rand_bkg_color=None):
        """Update the default values for invalid rays

        Args:
            output_valid: that contains keys with `rgb*`/`mask*`/`depth*`/`normal*`/`progress_*` in (N_valid, ...) shape
            For each keys, fill with
                rgb/rgb_*: by self.optim_params['bkg_color'], by default (1, 1, 1) as white bkg
            mask: mask indicating each rays' validity. in (N_rays, ), with B_valid `True` values.
            rand_bkg_color: the rand bkg color for training. By default None.

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
                    if rand_bkg_color is not None:
                        out_tensor = torch.ones(new_shape, dtype=dtype, device=device) * rand_bkg_color
                    else:
                        bkg_color = self.get_render_cfgs('bkg_color')
                        bkg_color = torch.tensor(bkg_color, dtype=dtype, device=device)[None]  # (B, 3)
                        out_tensor = torch.ones(new_shape, dtype=dtype, device=device) * bkg_color
                    out_tensor[mask] = v
                    output[k] = out_tensor
                elif k.startswith('depth'):  # update depth
                    depth_far = self.get_render_cfgs('depth_far')
                    depth_far = torch.tensor(depth_far, dtype=dtype, device=device)[None]  # (1, 1)
                    out_tensor = torch.ones(new_shape, dtype=dtype, device=device) * depth_far
                    out_tensor[mask] = v
                    output[k] = out_tensor
                elif k.startswith('mask'):  # update mask, must be 0
                    out_tensor = torch.zeros(new_shape, dtype=dtype, device=device)
                    out_tensor[mask] = v
                    output[k] = out_tensor
                elif k.startswith('normal'):  # update normal
                    normal = self.get_render_cfgs('normal')
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

    def optimize(self, cur_epoch=0):
        """Optimize the obj bounding geometric structure. Support ['volume'] now."""
        self.obj_bound.optimize(cur_epoch, self.get_n_coarse_sample(), self.get_est_opacity)

    def get_est_opacity(self, dt, pts):
        """Get the estimated opacity at certain pts. This method is only for fg_model.
        In density model, when density is high, opacity = 1 - exp(-sigma*dt), when sigma is large, opacity is large.
        You have to rewrite this function in sdf-like models

        For opacity calculation:
            - in instant-ngp, the opacity is used as `density * dt`,
            - you can also used `1.0 - torch.exp(-torch.relu(density) * dt)` as its real definition.

        Args:
            dt: the dt used for calculated
            pts: the pts in the field. (B, 3) xyz position. Need geometric model to process

        Returns:
            opacity: (B,) opacity. In density model, opacity = 1 - exp(-sigma*dt)
                                   For sdf model,  opacity = 1 - exp(-sdf_to_sigma(sdf)*dt)
            When opacity is large(Than some thresold), pts can be considered as in the object.
        """
        density = self.forward_pts(pts)  # (B,)
        opacity = density * dt  # (B,)

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
