# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.torch_utils import chunk_processing


class FullModel(nn.Module):
    """Full model for 3d reconstruction, combining foreground and background model.

     fg model has two type:
        (1) It contains one geo_net and one radiance_net, only produce one set of 'rgb/depth/mask' results.
        (2) It contains two stages and two sets of geo/radiance net, produce 'rgb_coarse/fine, etc' results.
            If fine stage does not exist, only get 'rgb_coarse'.
        Both coarse/fine stage are modelling in the foreground range.
     For the final output, only keep ones set of progress if get_progress=True.
         If inference only, output 'rgb' instead of 'rgb_coarse/fine'.
     """

    def __init__(self, cfgs, fg_model, bkg_cfgs=None, bkg_model=None):
        super(FullModel, self).__init__()
        self.cfgs = cfgs
        self.fg_model = fg_model
        self.bkg_cfgs = bkg_cfgs
        self.bkg_model = bkg_model
        # set and check bkg cfgs
        if self.bkg_cfgs is not None:
            # bkg_blend type, 'rgb' or 'sigma
            self.bkg_blend = get_value_from_cfgs_field(self.bkg_cfgs.model, 'bkg_blend', 'rgb')
            self.check_bkg_cfgs()
            # set fg_model add_inf_z=True to keep all sigma if 'sigma' mode
            if self.bkg_blend == 'sigma':
                self.fg_model.set_add_inf_z(True)

    def check_bkg_cfgs(self):
        """If bkg model is used, check for invalid cfgs"""
        if self.bkg_blend == 'rgb':
            assert self.fg_model.get_ray_cfgs('add_inf_z') is False, 'Do not add_inf_z for foreground'
            assert self.bkg_model.get_ray_cfgs('add_inf_z') is True,\
                'Must use add_inf_z for background in rgb blending mode'
        elif self.bkg_blend == 'sigma':
            assert self.bkg_model.get_ray_cfgs('add_inf_z') is False,\
                'Do not add_inf_z for background in sigma blending mode'
        else:
            raise NotImplementedError('Invalid bkg_blend type {}'.format(self.bkg_blend))

        # foreground far distance should not exceed 2*bkg_bounding_radius. foreground radius should be smaller as well.
        max_far = 2.0 * self.bkg_model.get_ray_cfgs('bounding_radius')
        fg_model_far = self.fg_model.get_ray_cfgs('far')
        if fg_model_far is None:
            if self.fg_model.get_ray_cfgs('bounding_radius') is not None:
                assert self.fg_model.get_ray_cfgs('bounding_radius') <= self.bkg_model.get_ray_cfgs('bounding_radius'),\
                    'fg_model radius should not exceed bkg_model radius'
        else:
            assert fg_model_far is None or fg_model_far <= max_far,\
                'Do not set fg_model far exceed {}'.format(max_far)

    def get_fg_model(self):
        """Get the foreground model"""
        return self.fg_model

    def get_bkg_model(self):
        """Get the background model"""
        return self.bkg_model

    def get_chunk_rays(self):
        """Get the chunk rays num from fg_model"""
        return self.fg_model.get_chunk_rays()

    def get_chunk_pts(self):
        """Get the chunk pts num from fg_model"""
        return self.fg_model.get_chunk_pts()

    def set_chunk_rays(self, chunk_rays):
        """Set the chunk rays num for both model"""
        self.fg_model.set_chunk_rays(chunk_rays)
        if self.bkg_model is not None:
            self.bkg_model.set_chunk_rays(chunk_rays)

    def set_chunk_pts(self, chunk_pts):
        """Set the chunk pts num for both model"""
        self.fg_model.set_chunk_pts(chunk_pts)
        if self.bkg_model is not None:
            self.bkg_model.set_chunk_pts(chunk_pts)

    def pretrain_siren(self):
        """Pretrain siren layer of implicit network for both models."""
        self.fg_model.pretrain_siren()
        if self.bkg_model is not None:
            self.bkg_model.pretrain_siren()

    def is_cuda(self):
        """Check whether the model is on cuda"""
        return next(self.parameters()).is_cuda

    def sigma_reverse(self):
        """Whether fg_model's implicit model is modeling sigma or sdf(flow different)"""
        return self.fg_model.sigma_reverse()

    @staticmethod
    def clean_two_stage_progress(output):
        """Keep one set of progress in output. """
        if not any([k.startswith('progress_') for k in output.keys()]):
            return output

        progress_keys = [k for k in output.keys() if k.startswith('progress_')]
        if any([(not k.endswith('_coarse') and not k.endswith('_fine'))
                for k in progress_keys]):  # contains one stage result
            pop_keys = [k for k in progress_keys if k.endswith('_coarse') or k.endswith('_fine')]
            for k in pop_keys:
                output.pop(k)
        elif any([k.endswith('_fine') for k in progress_keys]):  # contains fine result
            pop_keys = [k for k in progress_keys if k.endswith('_coarse')]  # pop coarse result
            for k in pop_keys:
                output.pop(k)
            rewrite_keys = [k for k in progress_keys if k.endswith('_fine')]  # rewrite fine result
            for k in rewrite_keys:
                output[k.replace('_fine', '')] = output[k]
                output.pop(k)
        else:  # only coarse result
            rewrite_keys = [k for k in progress_keys if k.endswith('_coarse')]  # rewrite coarse result
            for k in rewrite_keys:
                output[k.replace('_coarse', '')] = output[k]
                output.pop(k)

        return output

    @staticmethod
    def clean_progress(output):
        """Remove all keys with progress_"""
        progress_keys = [k for k in output.keys() if k.startswith('progress_')]
        for k in progress_keys:
            output.pop(k)

        return output

    @staticmethod
    def detach_progress(output):
        """Detach progress to delete graph"""
        progress_keys = [k for k in output.keys() if k.startswith('progress_')]
        for k in progress_keys:
            if isinstance(output[k], torch.Tensor):
                output[k] = output[k].detach()

        return output

    def blend_two_stage_bkg_sigma(self, fg_output, bkg_output, inference_only=False, get_progress=False):
        """blend fg + bkg for sigma/radiance with coarse/fine output and re-run ray marching together.
        You must make sure that the sigma can be merged together. Otherwise do not use it(Like sdf method).
        All inputs flatten in (B, x) dim
        """
        assert 'progress_sigma_coarse' in fg_output, 'You must get_progress for fg_model'
        # coarse stage output from fg_model, (B, n_fg + n_bkg-1), already sorted since fg/bkg in different range
        sigma_coarse_all = torch.cat([fg_output['progress_sigma_coarse'], bkg_output['progress_sigma']], 1)
        radiance_coarse_all = torch.cat([fg_output['progress_radiance_coarse'], bkg_output['progress_radiance']], 1)
        zvals_coarse_all = torch.cat([fg_output['progress_zvals_coarse'], bkg_output['progress_zvals']], 1)

        # re-run fg ray-marching in coarse stage
        fg_output_coarse = self.fg_model.ray_marching(
            sigma_coarse_all,
            radiance_coarse_all,
            zvals_coarse_all,
            add_inf_z=self.fg_model.get_ray_cfgs('add_inf_z'),
            inference_only=inference_only
        )

        # just replace overall rgb and depth
        for key in ['rgb', 'depth']:
            fg_output[key + '_coarse'] = fg_output_coarse[key]

        # fine stage output from fg_model, (B, n_fg + n_bkg-1), already sorted since fg/bkg in different range
        if 'progress_sigma_fine' in fg_output:
            # (B, n_fg + n_bkg-1), already sorted since fg/bkg in different range
            sigma_fine_all = torch.cat([fg_output['progress_sigma_fine'], bkg_output['progress_sigma']], 1)
            radiance_fine_all = torch.cat([fg_output['progress_radiance_fine'], bkg_output['progress_radiance']], 1)
            zvals_fine_all = torch.cat([fg_output['progress_zvals_fine'], bkg_output['progress_zvals']], 1)

            # re-run fg ray-marching in fine stage
            fg_output_fine = self.fg_model.ray_marching(
                sigma_fine_all,
                radiance_fine_all,
                zvals_fine_all,
                add_inf_z=self.fg_model.get_ray_cfgs('add_inf_z'),
                inference_only=inference_only
            )

            # just replace overall rgb and depth
            for key in ['rgb', 'depth']:
                fg_output[key + '_fine'] = fg_output_fine[key]

        # keep one set of progress
        fg_output = self.clean_two_stage_progress(fg_output)

        return fg_output

    def blend_bkg_sigma(self, fg_output, bkg_output, inference_only=False, get_progress=False):
        """blend fg + bkg for sigma/radiance and re-run ray marching together.
        You must make sure that the sigma can be merged together. Otherwise do not use it(Like sdf method).
        All inputs flatten in (B, x) dim
        """
        if any([key.endswith('_coarse') or key.endswith('_fine') for key in fg_output.keys()]):
            return self.blend_two_stage_bkg_sigma(fg_output, bkg_output, inference_only, get_progress)

        assert 'progress_sigma' in fg_output, 'You must get_progress for fg_model'

        # (B, n_fg + n_bkg-1), already sorted since fg/bkg in different range
        sigma_all = torch.cat([fg_output['progress_sigma'], bkg_output['progress_sigma']], 1)
        radiance_all = torch.cat([fg_output['progress_radiance'], bkg_output['progress_radiance']], 1)
        zvals_all = torch.cat([fg_output['progress_zvals'], bkg_output['progress_zvals']], 1)

        # re-run fg ray-marching
        fg_output_all = self.fg_model.ray_marching(
            sigma_all,
            radiance_all,
            zvals_all,
            add_inf_z=self.fg_model.get_ray_cfgs('add_inf_z'),
            inference_only=inference_only
        )

        # just replace overall rgb and depth
        for key in ['rgb', 'depth']:
            fg_output[key] = fg_output_all[key]

        return fg_output

    def blend_two_stage_bkg_rgb(self, fg_output, bkg_output):
        """ blend fg + bkg for rgb and depth with coarse/fine output. mask is still for foreground only.
        All inputs flatten in (B, x) dim
        """
        assert 'progress_trans_shift_coarse' in fg_output, 'You must get_progress for fg_model'

        bkg_lamba_coarse = fg_output['progress_trans_shift_coarse'][:, -1]  # (B,) prob that light pass foreground
        fg_output['rgb_coarse'] = fg_output['rgb_coarse'] + bkg_lamba_coarse[:, None] * bkg_output['rgb']
        fg_output['depth_coarse'] = fg_output['depth_coarse'] + bkg_lamba_coarse * bkg_output['depth']
        if 'rgb_fine' in fg_output:
            bkg_lamba_fine = fg_output['progress_trans_shift_fine'][:, -1]  # (B,) prob that light pass foreground
            fg_output['rgb_fine'] = fg_output['rgb_fine'] + bkg_lamba_fine[:, None] * bkg_output['rgb']
            fg_output['depth_fine'] = fg_output['depth_fine'] + bkg_lamba_fine * bkg_output['depth']

        # keep one set of progress
        fg_output = self.clean_two_stage_progress(fg_output)

        return fg_output

    def blend_bkg_rgb(self, fg_output, bkg_output):
        """ blend fg + bkg for rgb and depth. mask is still for foreground only.
        All inputs flatten in (B, x) dim
        """
        if any([key.endswith('_coarse') or key.endswith('_fine') for key in fg_output.keys()]):
            return self.blend_two_stage_bkg_rgb(fg_output, bkg_output)

        assert 'progress_trans_shift' in fg_output, 'You must get_progress for fg_model'
        bkg_lamba = fg_output['progress_trans_shift'][:, -1]  # (B,) prob that light passed through foreground
        fg_output['rgb'] = fg_output['rgb'] + bkg_lamba[:, None] * bkg_output['rgb']
        fg_output['depth'] = fg_output['depth'] + bkg_lamba * bkg_output['depth']

        return fg_output

    def blend_output(self, fg_output, bkg_output=None, inference_only=False, get_progress=False):
        """Blend output based on bkg_blend type"""
        if bkg_output is None:
            final_output = self.clean_two_stage_progress(fg_output)
        else:
            if self.bkg_blend == 'rgb':
                final_output = self.blend_bkg_rgb(fg_output, bkg_output)
            elif self.bkg_blend == 'sigma':
                final_output = self.blend_bkg_sigma(fg_output, bkg_output, inference_only, get_progress)
            else:
                raise NotImplementedError('Invalid bkg_blend type {}...'.format(self.bkg_blend))

        # clean progress
        if not get_progress:
            final_output = self.clean_progress(final_output)

        return final_output

    def blend_rand_bkg_color(self, inputs, flat_inputs, inference_only):
        """Blend random bkg color to the rgb inputs"""
        dtype = inputs['img'].dtype
        device = inputs['img'].device
        batch_size, n_rays_per_batch = inputs['img'].shape[:2]

        rand_bkg_color = self.fg_model.get_ray_cfgs('rand_bkg_color')
        if rand_bkg_color and not inference_only and flat_inputs['mask'] is not None:
            mask = flat_inputs['mask']
            rand_bkg_color = torch.rand((batch_size * n_rays_per_batch, 3), dtype=dtype,
                                        device=device).detach()  # (BN, 3)
            flat_inputs['rand_bkg_color'] = rand_bkg_color.clone()
            flat_inputs['img'] = flat_inputs['img'] * mask[:, None] + rand_bkg_color * (1.0 - mask[:, None])
            # modify original inputs for loss calculation as well
            inputs['img'] = flat_inputs['img'].view(batch_size, n_rays_per_batch, 3)
        else:
            flat_inputs['rand_bkg_color'] = None

    def prepare_flatten_inputs(self, inputs, inference_only=False):
        """Prepare the inputs by flatten them from (B, N, ...) to (BN, ...)

        Args:
            inputs: a dict of torch tensor:
                inputs['img']: torch.tensor (B, N, 3), rgb image color in (0, 1)
                inputs['rays_o']: torch.tensor (B, N, 3), cam_loc/ray_start position
                inputs['rays_d']: torch.tensor (B, N, 3), view dir(assume normed)
                inputs['rays_r']: torch.tensor (B, N, 1), radius
                inputs['mask']: torch.tensor (B, N), mask value in {0, 1}. optional
                inputs['bounds']: torch.tensor (B, N, 2). optional
            inference_only: this device whether to use the random bkg_color

        Returns:
            flatten_inputs:
                value in inputs flatten into (BN, ...)
        """
        flat_inputs = {}
        img = inputs['img'].view(-1, 3)  # (BN, 3)
        rays_o = inputs['rays_o'].view(-1, 3)  # (BN, 3)
        rays_d = inputs['rays_d'].view(-1, 3)  # (BN, 3)
        rays_r = inputs['rays_r'].view(-1, 1)  # (BN, 1)
        batch_size, n_rays_per_batch = inputs['rays_o'].shape[:2]

        flat_inputs['img'] = img
        flat_inputs['rays_o'] = rays_o
        flat_inputs['rays_d'] = rays_d
        flat_inputs['rays_r'] = rays_r

        # optional inputs
        bounds = None
        if 'bounds' in inputs:
            bounds = inputs['bounds'].view(-1, 2)  # (BN, 2)
        flat_inputs['bounds'] = bounds

        mask = None
        if 'mask' in inputs:
            mask = inputs['mask'].view(-1)  # (BN,)
        flat_inputs['mask'] = mask

        # blend random bkg color
        self.blend_rand_bkg_color(inputs, flat_inputs, inference_only)

        return flat_inputs, batch_size, n_rays_per_batch

    @staticmethod
    def reshape_output(output, batch_size, n_rays_per_batch):
        """Reshape flatten output from (BN, ...) into (B, N, ...) dim"""
        for k, v in output.items():
            if isinstance(v, torch.Tensor) and batch_size * n_rays_per_batch == v.shape[0]:
                new_shape = tuple([batch_size, n_rays_per_batch] + list(v.shape)[1:])
                output[k] = v.view(new_shape)
            else:
                output[k] = v

        return output

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        """Do not call forward directly using chunk_process since the tensor are not flatten to represent batch of rays.
        It will call fg/bkg model.forward() with flatten inputs, and blend the result if bkg_model exists.

        Args:
            inputs: a dict of torch tensor:
                inputs['img']: torch.tensor (B, N, 3), rgb image color in (0, 1)
                inputs['rays_o']: torch.tensor (B, N, 3), cam_loc/ray_start position
                inputs['rays_d']: torch.tensor (B, N, 3), view dir(assume normed)
                inputs['rays_r']: torch.tensor (B, N, 1), radius
                inputs['mask']: torch.tensor (B, N), mask value in {0, 1}. optional
                inputs['bounds']: torch.tensor (B, N, 2). optional
            inference_only: If True, only return the final results(not coarse, no progress).
                            Use in eval/infer mode to save memory. By default False
            get_progress: If True, output some progress for recording(in foreground),
                          can not used in inference only mode. By default False
            cur_epoch: current epoch, for training purpose only. By default 0.
            total_epoch: total num of epoch, for training purpose only. By default 300k.

        Returns:
            output: is a dict keys like (rgb/rgb_coarse/rgb_fine, depth, mask, normal, etc) based on _forward function.
                      in (B, N, ...) dim.
            If get_progress is True, output will contain keys like 'progress_xx' for xx in ['sigma', 'zvals', etc].
        """
        # prepare flatten inputs
        flat_inputs, batch_size, n_rays_per_batch = self.prepare_flatten_inputs(inputs, inference_only)

        get_progress_fg = True if (self.bkg_model is not None) else get_progress  # need the progress to blend
        fg_output = chunk_processing(
            self.fg_model.forward, self.fg_model.get_chunk_rays(), False, flat_inputs, inference_only, get_progress_fg,
            cur_epoch, total_epoch
        )

        bkg_output = None
        if self.bkg_model is not None:
            bkg_output = chunk_processing(
                self.bkg_model.forward, self.bkg_model.get_chunk_rays(), False, flat_inputs, inference_only, True,
                cur_epoch, total_epoch
            )  # bkg model always keep progress item for blending. Will not be saved after merge

        # merge output and detach progress item
        output = self.blend_output(fg_output, bkg_output, inference_only, get_progress)
        output = self.detach_progress(output)

        # reshape values from (B*N, ...) to (B, N, ...)
        output = self.reshape_output(output, batch_size, n_rays_per_batch)

        return output

    def surface_render(
        self,
        inputs,
        method='sphere_tracing',
        n_step=128,
        n_iter=100,
        threshold=0.01,
        level=0.0,
        grad_dir='ascent',
        **kwargs
    ):
        """Surface rendering using foreground model. Only in inference mode.

        Args:
            inputs: a dict of torch tensor:
                inputs['img']: torch.tensor (B, N, 3), rgb image color in (0, 1)
                inputs['rays_o']: torch.tensor (B, N, 3), cam_loc/ray_start position
                inputs['rays_d']: torch.tensor (B, N, 3), view dir(assume normed)
                inputs['mask']: torch.tensor (B, N), mask value in {0, 1}. optional
                inputs['bounds']: torch.tensor (B, N, 2). optional
            method: method used to find the intersection. support
                ['sphere_tracing', 'secant_root_finding']
            n_step: used for secant_root_finding, split the whole ray into intervals. By default 128
            n_iter: num of iter to run finding algorithm. By default 100, large enough to escape
            threshold: error bounding to stop the iteration. By default 0.01
            level: the surface pts geo_value offset. 0.0 is for sdf. some positive value may be for density.
            grad_dir: If descent, the inner obj has geo_value > level,
                                find the root where geo_value first meet ---level+++
                      If ascent, the inner obj has geo_value < level(like sdf),
                                find the root where geo_value first meet +++level---

        Returns:
            output: is a dict keys like (rgb/rgb_coarse/rgb_fine, depth, mask, normal, etc) based on _forward function.
                    in (B, N, ...) dim.
        """
        # prepare flatten inputs
        flat_inputs, batch_size, n_rays_per_batch = self.prepare_flatten_inputs(inputs)

        # fg_model do surface tracing
        output = chunk_processing(
            self.fg_model.surface_render, self.fg_model.get_chunk_rays(), False, flat_inputs, method, n_step, n_iter,
            threshold, level, grad_dir
        )

        # reshape values from (B*N, ...) to (B, N, ...)
        output = self.reshape_output(output, batch_size, n_rays_per_batch)

        return output

    @torch.no_grad()
    def optimize(self, cur_epoch=0):
        """Optimize the fg_model for its obj_bound structure."""
        self.fg_model.optimize(cur_epoch)

    def forward_pts_dir(self, pts: torch.Tensor, view_dir: torch.Tensor = None):
        """Only the fg model can forward pts and dir"""
        return self.fg_model.forward_pts_dir(pts, view_dir)

    def forward_pts(self, pts: torch.Tensor):
        """Only the fg model can forward pts"""
        return self.fg_model.forward_pts(pts)

    def get_est_opacity(self, dt, pts: torch.Tensor):
        """Only the fg model can forward pts"""
        return self.fg_model.get_est_opacity(dt, pts)
