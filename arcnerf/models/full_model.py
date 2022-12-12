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
        self.fg_only = False
        if self.bkg_cfgs is not None:
            # bkg_blend type, 'rgb' or 'sigma
            self.bkg_blend = get_value_from_cfgs_field(self.bkg_cfgs.model, 'bkg_blend', 'rgb')
            self.check_bkg_cfgs()
            # set fg_model add_inf_z=True to keep all sigma if 'sigma' mode
            if self.bkg_blend == 'sigma':
                self.fg_model.set_add_inf_z(True)

            # whether to render fg only
            self.fg_only = get_value_from_cfgs_field(self.bkg_cfgs.model, 'fg_only', False)

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

    def init_setting(self):
        """Init the setting like pretrain siren layers."""
        self.fg_model.init_setting()
        if self.bkg_model is not None:
            self.bkg_model.init_setting()

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

        def run_blend_sigma(
            fg_output, bkg_output, fg_key='_coarse', bkg_key='', inference_only=False, get_progress=False
        ):
            # reset the invalid sample, such that fg_output is sampling after some bkg sample
            zvals_fg = fg_output['progress_zvals{}'.format(fg_key)]
            zvals_bkg = bkg_output['progress_zvals{}'.format(bkg_key)]
            invalid_idx = zvals_fg[:, -1] > zvals_bkg[:, 0]
            # reset to 0
            sigma_fg = fg_output['progress_sigma{}'.format(fg_key)]
            sigma_bkg = bkg_output['progress_sigma{}'.format(bkg_key)]
            radiance_fg = fg_output['progress_radiance{}'.format(fg_key)]
            radiance_bkg = bkg_output['progress_radiance{}'.format(bkg_key)]
            sigma_fg[invalid_idx] = 0
            radiance_fg[invalid_idx] = 0
            zvals_fg[invalid_idx] = 0

            # coarse stage output from fg_model, (B, n_fg + n_bkg-1), already sorted since fg/bkg in different range
            sigma_all = torch.cat([sigma_fg, sigma_bkg], 1)
            radiance_all = torch.cat([radiance_fg, radiance_bkg], 1)
            zvals_all = torch.cat([zvals_fg, zvals_bkg], 1)

            # re-run fg ray-marching in coarse stage
            fg_output_all = self.fg_model.ray_marching(
                sigma_all, radiance_all, zvals_all, inference_only=inference_only
            )

            # get progress for fg_model only
            fg_output_all = self.fg_model.output_get_progress(fg_output_all, get_progress, sigma_fg.shape[1])

            # replace the keys
            final_out = {}
            for k, v in fg_output_all.items():
                if k == 'mask' and k + fg_key in fg_output.keys():  # The mask is still from fg output only
                    final_out[k + fg_key] = fg_output[k + fg_key]
                else:
                    final_out[k + fg_key] = v

            return final_out

        if any([key.endswith('_coarse') or key.endswith('_fine') for key in bkg_output.keys()]):  # bkg is two stage
            blend_coarse_output = run_blend_sigma(
                fg_output, bkg_output, '_coarse', '_coarse', inference_only, get_progress
            )
            blend_fine_output = {}
            if 'progress_sigma_fine' in fg_output:
                if 'progress_sigma_fine' in bkg_output:
                    blend_fine_output = run_blend_sigma(
                        fg_output, bkg_output, '_fine', '_fine', inference_only, get_progress
                    )
                else:
                    blend_fine_output = run_blend_sigma(
                        fg_output, bkg_output, '_fine', '_coarse', inference_only, get_progress
                    )
        else:  # bkg is one stage
            blend_coarse_output = run_blend_sigma(fg_output, bkg_output, '_coarse', '', inference_only, get_progress)
            blend_fine_output = {}
            if 'progress_sigma_fine' in fg_output:
                blend_fine_output = run_blend_sigma(fg_output, bkg_output, '_fine', '', inference_only, get_progress)

        # merge two stage
        blend_out = {}
        for k, v in blend_coarse_output.items():
            blend_out[k] = v
        for k, v in blend_fine_output.items():
            blend_out[k] = v

        # keep one set of progress
        blend_out = self.clean_two_stage_progress(blend_out)

        return blend_out

    def blend_bkg_sigma(self, fg_output, bkg_output, inference_only=False, get_progress=False):
        """blend fg + bkg for sigma/radiance and re-run ray marching together.
        You must make sure that the sigma can be merged together. Otherwise do not use it(Like sdf method).
        All inputs flatten in (B, x) dim

        NOTICE: We don't suggest blend sigma since it does not have advantage over rgb mode. And sometime
        the representation of inner/outer sigma are not the same (eg. sdf + density).
        Not full functionality is provided for sigma mode.
        """
        if any([key.endswith('_coarse') or key.endswith('_fine') for key in fg_output.keys()]):
            return self.blend_two_stage_bkg_sigma(fg_output, bkg_output, inference_only, get_progress)

        assert 'progress_sigma' in fg_output, 'You must get_progress for fg_model'
        if any([key.endswith('_coarse') or key.endswith('_fine') for key in bkg_output.keys()]):  # bkg is two stage
            if 'progress_sigma_fine' in bkg_output.keys():
                bkg_sigma = bkg_output['progress_sigma_fine']
                bkg_radiance = bkg_output['progress_radiance_fine']
                bkg_zvals = bkg_output['progress_zvals_fine']
            else:
                bkg_sigma = bkg_output['progress_sigma_coarse']
                bkg_radiance = bkg_output['progress_radiance_coarse']
                bkg_zvals = bkg_output['progress_zvals_coarse']
        else:
            bkg_sigma = bkg_output['progress_sigma']
            bkg_radiance = bkg_output['progress_radiance']
            bkg_zvals = bkg_output['progress_zvals']

        # reset the invalid sample, such that fg_output is sampling after some bkg sample
        fg_zvals = fg_output['progress_zvals']
        invalid_idx = fg_zvals[:, -1] > bkg_zvals[:, 0]
        # reset to 0
        fg_sigma = fg_output['progress_sigma']
        fg_radiance = fg_output['progress_radiance']
        fg_sigma[invalid_idx] = 0
        fg_radiance[invalid_idx] = 0
        fg_zvals[invalid_idx] = 0

        # (B, n_fg + n_bkg-1), already sorted since fg/bkg in different range
        sigma_all = torch.cat([fg_sigma, bkg_sigma], 1)
        radiance_all = torch.cat([fg_radiance, bkg_radiance], 1)
        zvals_all = torch.cat([fg_zvals, bkg_zvals], 1)

        # re-run fg ray-marching
        fg_output_all = self.fg_model.ray_marching(sigma_all, radiance_all, zvals_all, inference_only=inference_only)

        # get progress for fg_model only
        fg_output_all = self.fg_model.output_get_progress(fg_output_all, get_progress, fg_sigma.shape[1])

        # replace the keys
        final_out = {}
        for k, v in fg_output_all.items():
            if k == 'mask' and k in fg_output.keys():  # The mask is still from fg output only
                final_out[k] = fg_output[k]
            else:
                final_out[k] = v

        return fg_output_all

    def blend_two_stage_bkg_rgb(self, fg_output, bkg_output):
        """ blend fg + bkg for rgb and depth with coarse/fine output. mask is still for foreground only.
        All inputs flatten in (B, x) dim
        """
        assert 'progress_trans_shift_coarse' in fg_output, 'You must get_progress for fg_model'

        if any([key.endswith('_coarse') or key.endswith('_fine') for key in bkg_output.keys()]):  # bkg is two stage
            bkg_lamba_coarse = fg_output['progress_trans_shift_coarse'][:, -1]  # (B,) prob that light pass foreground
            fg_output['rgb_coarse'] = fg_output['rgb_coarse'] + bkg_lamba_coarse[:, None] * bkg_output['rgb_coarse']
            fg_output['depth_coarse'] = fg_output['depth_coarse'] + bkg_lamba_coarse * bkg_output['depth_coarse']
            if 'rgb_fine' in fg_output:
                bkg_lamba_fine = fg_output['progress_trans_shift_fine'][:, -1]
                if 'rgb_fine' in bkg_output:  # merge with fine bkg
                    fg_output['rgb_fine'] = fg_output['rgb_fine'] + bkg_lamba_fine[:, None] * bkg_output['rgb_fine']
                    fg_output['depth_fine'] = fg_output['depth_fine'] + bkg_lamba_fine * bkg_output['depth_fine']
                else:  # merge with coarse bkg
                    fg_output['rgb_fine'] = fg_output['rgb_fine'] + bkg_lamba_fine[:, None] * bkg_output['rgb_coarse']
                    fg_output['depth_fine'] = fg_output['depth_fine'] + bkg_lamba_fine * bkg_output['depth_coarse']
        else:  # bkg one stage
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
        if any([key.endswith('_coarse') or key.endswith('_fine') for key in bkg_output.keys()]):  # bkg is two stage
            if 'rgb_fine' in bkg_output:  # merge with fine bkg
                fg_output['rgb'] = fg_output['rgb'] + bkg_lamba[:, None] * bkg_output['rgb_fine']
                fg_output['depth'] = fg_output['depth'] + bkg_lamba * bkg_output['depth_fine']
            else:  # merge with coarse bkg
                fg_output['rgb'] = fg_output['rgb'] + bkg_lamba[:, None] * bkg_output['rgb_coarse']
                fg_output['depth'] = fg_output['depth'] + bkg_lamba * bkg_output['depth_coarse']
        else:  # bkg is one stage
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

    def prepare_flatten_inputs(self, inputs):
        """Prepare the inputs by flatten them from (B, N, ...) to (BN, ...)

        Args:
            inputs: a dict of torch tensor:
                inputs['img']: torch.tensor (B, N, 3), rgb image color in (0, 1), optional
                inputs['rays_o']: torch.tensor (B, N, 3), cam_loc/ray_start position
                inputs['rays_d']: torch.tensor (B, N, 3), view dir(assume normed)
                inputs['rays_r']: torch.tensor (B, N, 1), radius
                inputs['mask']: torch.tensor (B, N), mask value in {0, 1}. optional
                inputs['bounds']: torch.tensor (B, N, 2). zvals near/far bound, optional
                inputs['bkg_color']: torch.tensor (B, N, 3), random/fix bkg color, optional

        Returns:
            flatten_inputs:
                value in inputs flatten into (BN, ...)
        """
        flat_inputs = {}
        img = inputs['img'].view(-1, 3) if 'img' in inputs.keys() else None  # (BN, 3)
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

        bkg_color = None
        if 'bkg_color' in inputs:
            bkg_color = inputs['bkg_color'].view(-1, 3)  # (BN,)
        flat_inputs['bkg_color'] = bkg_color

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
                inputs['img']: torch.tensor (B, N, 3), rgb image color in (0, 1), optional
                inputs['rays_o']: torch.tensor (B, N, 3), cam_loc/ray_start position
                inputs['rays_d']: torch.tensor (B, N, 3), view dir(assume normed)
                inputs['rays_r']: torch.tensor (B, N, 1), radius
                inputs['mask']: torch.tensor (B, N), mask value in {0, 1}. optional
                inputs['bounds']: torch.tensor (B, N, 2). zvals near/far bound, optional
                inputs['bkg_color']: torch.tensor (B, N, 3), random/fix bkg color, optional
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
        flat_inputs, batch_size, n_rays_per_batch = self.prepare_flatten_inputs(inputs)

        # chunk process fg+bkg function and merge result
        chunk_rays = self.fg_model.get_chunk_rays()
        if self.bkg_model is not None:
            chunk_rays = min(self.fg_model.get_chunk_rays(), self.bkg_model.get_chunk_rays())

        output = chunk_processing(
            self.process_fg_bkg_model, chunk_rays, False, self.fg_model, self.bkg_model, flat_inputs, inference_only,
            get_progress, cur_epoch, total_epoch
        )

        # reshape values from (B*N, ...) to (B, N, ...)
        output = self.reshape_output(output, batch_size, n_rays_per_batch)

        return output

    def process_fg_bkg_model(
        self, fg_model, bkg_model, flat_inputs, inference_only, get_progress, cur_epoch, total_epoch
    ):
        """In case accumulate too many progress output during chunk_process, directly merge the output"""
        get_progress_fg = True if (bkg_model is not None) else get_progress  # need the progress to blend
        fg_output = fg_model.forward(flat_inputs, inference_only, get_progress_fg, cur_epoch, total_epoch)

        # bkg model always keep progress item for blending. Will not be saved after merge
        bkg_output = None
        if bkg_model is not None and not self.fg_only:
            bkg_output = bkg_model.forward(flat_inputs, inference_only, True, cur_epoch, total_epoch)

        # merge output and detach progress item
        output = self.blend_output(fg_output, bkg_output, inference_only, get_progress)
        output = self.detach_progress(output)

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
        """Optimize the fg_model/bkg_model for its obj_bound structure."""
        self.fg_model.optimize(cur_epoch)
        if self.bkg_model is not None:
            self.bkg_model.optimize(cur_epoch)

    def forward_pts_dir(self, pts: torch.Tensor, view_dir: torch.Tensor = None):
        """Only the fg model can forward pts and dir"""
        return self.fg_model.forward_pts_dir(pts, view_dir)

    def forward_pts(self, pts: torch.Tensor):
        """Only the fg model can forward pts"""
        return self.fg_model.forward_pts(pts)

    def get_est_opacity(self, dt, pts: torch.Tensor):
        """Only the fg model can forward pts"""
        return self.fg_model.get_est_opacity(dt, pts)

    def get_dynamicbs_factor(self):
        """Get the dynamic factor from fg model"""
        return self.fg_model.get_dynamicbs_factor()

    def reset_measurement(self):
        """Reset the measurement for dynamic batchsize from fg model"""
        self.fg_model.reset_measurement()
