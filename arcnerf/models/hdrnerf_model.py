# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from arcnerf.models.base_modules import get_activation, DenseLayer
from common.utils.cfgs_utils import dict_to_obj, get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from .nerf_model import NeRF


@MODEL_REGISTRY.register()
class HDRNeRF(NeRF):
    """ HDRNerf model. It use the same structure as NeRF, but add several tiny mlps
        ref: https://www.matthewtancik.com/nerf
    """

    def __init__(self, cfgs):
        super(HDRNeRF, self).__init__(cfgs)
        model_cfgs = self.cfgs.model
        # build tiny mlps for exp_time
        self.coarse_exp_r_mlps, self.coarse_exp_g_mlps, self.coarse_exp_b_mlps = \
            self.build_exp_mlps(model_cfgs.exp_mlps)
        # set fine model if n_importance > 0:
        if self.get_ray_cfgs('n_importance') > 0:
            if self.get_ray_cfgs('shared_network'):  # use the same network
                self.fine_exp_r_mlps, self.fine_exp_g_mlps, self.fine_exp_b_mlps = \
                    self.coarse_exp_r_mlps, self.coarse_exp_g_mlps, self.coarse_exp_b_mlps
            else:  # separate network
                self.fine_exp_r_mlps, self.fine_exp_g_mlps, self.fine_exp_b_mlps = \
                    self.build_exp_mlps(model_cfgs.exp_mlps)

    def build_exp_mlps(self, cfgs):
        """Build the separate mlps for rgb_h to rgb_l"""
        act_cfgs = get_value_from_cfgs_field(cfgs, 'act_cfgs', None)
        out_act_cfgs = get_value_from_cfgs_field(cfgs, 'out_act_cfg', None)

        sep_layers = []
        for _ in range(3):
            layers = []
            for i in range(cfgs.D + 1):
                in_dim = 1 if i == 0 else cfgs.W
                out_dim = 1 if i == cfgs.D else cfgs.W

                if i != cfgs.D:
                    layer = DenseLayer(in_dim, out_dim, activation=get_activation(act_cfgs))
                else:
                    sigmoid_cfg = dict_to_obj({'type': 'Sigmoid'})
                    layer = DenseLayer(in_dim, out_dim, activation=get_activation(out_act_cfgs, sigmoid_cfg))

                layers.append(layer)

            sep_layers.append(nn.ModuleList(layers))

        return sep_layers[0], sep_layers[1], sep_layers[2]

    def forward_exp_mlps(self, l_r, l_g, l_b, rgb_h, exp_time):
        """Forward the tiny mlps with rgb_h as inputs

        Args:
            l_r, l_g, l_b: layers for r/g/b color from (B,) to (B,)
            rgb_h: hdr value (not exponential) in (B, 3)
            exp_time: exposure time in (B,)

        Return:
            rgb_l: LDR value in (B, 3)
        """
        r_h_s = (rgb_h[:, 0] + torch.log(exp_time))[:, None]  # (B, 1)
        g_h_s = (rgb_h[:, 1] + torch.log(exp_time))[:, None]  # (B, 1)
        b_h_s = (rgb_h[:, 2] + torch.log(exp_time))[:, None]  # (B, 1)

        for layer in l_r:
            r_h_s = layer(r_h_s)  # (B, 1)
        for layer in l_g:
            g_h_s = layer(g_h_s)  # (B, 1)
        for layer in l_b:
            b_h_s = layer(b_h_s)  # (B, 1)

        rgb_l = torch.cat([r_h_s, g_h_s, b_h_s], -1)  # (B, 3)

        return rgb_l

    def _forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        """Compare to original nerf, hdr nerf model the exp_time with tiny mlps"""
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        zvals = inputs['zvals']  # (B, n_pts)
        mask_pts = inputs['mask_pts']  # (B, n_pts)
        bkg_color = inputs['bkg_color']  # (B, 3)
        exp_time = inputs['exp_time']  # (B, )
        output = {}

        # get coarse pts sigma/rgb  (B, N_sample, ...)
        sigma, rgb_h = self.get_sigma_radiance_by_mask_pts(
            self.coarse_geo_net, self.coarse_radiance_net, rays_o, rays_d, zvals, mask_pts, inference_only
        )

        # do ldr calculation
        exp_time_repeat = torch.repeat_interleave(exp_time, rgb_h.shape[1], 0)
        rgb_l = self.forward_exp_mlps(
            self.coarse_exp_r_mlps, self.coarse_exp_g_mlps, self.coarse_exp_b_mlps, rgb_h.view(-1, 3), exp_time_repeat
        ).view(rays_o.shape[0], -1, 3)

        # ray marching for coarse network, keep the coarse weights for next stage
        output_coarse = self.ray_marching(sigma, rgb_l, zvals, inference_only=inference_only, bkg_color=bkg_color)
        coarse_weights = output_coarse['weights']
        # Get the marching hdr value
        if 'rgb' in output_coarse.keys():
            output_coarse['hdr'] = self.ray_marching(
                sigma, torch.exp(rgb_h), zvals, inference_only=inference_only, bkg_color=bkg_color
            )['rgb']
        # for unit loss
        if not inference_only:
            output_coarse['unit_exp'] = self.point_constraint(
                self.coarse_exp_r_mlps, self.coarse_exp_g_mlps, self.coarse_exp_b_mlps
            )

        # handle progress
        output['coarse'] = self.output_get_progress(output_coarse, get_progress)

        # fine model
        if self.get_ray_cfgs('n_importance') > 0:
            # get upsampled zvals
            zvals, mask_pts = self.upsample_zvals(zvals, coarse_weights, mask_pts, inference_only)

            # get upsampled pts sigma/rgb  (B, N_total, ...)
            sigma, rgb_h = self.get_sigma_radiance_by_mask_pts(
                self.fine_geo_net, self.fine_radiance_net, rays_o, rays_d, zvals, mask_pts, inference_only
            )

            # do ldr calculation
            exp_time_repeat = torch.repeat_interleave(exp_time, rgb_h.shape[1], 0)
            rgb_l = self.forward_exp_mlps(
                self.fine_exp_r_mlps, self.fine_exp_g_mlps, self.fine_exp_b_mlps, rgb_h.view(-1, 3), exp_time_repeat
            ).view(rays_o.shape[0], -1, 3)

            # ray marching for fine network
            output_fine = self.ray_marching(sigma, rgb_l, zvals, inference_only=inference_only, bkg_color=bkg_color)
            # Get the marching hdr value
            if 'rgb' in output_fine.keys():
                output_fine['hdr'] = self.ray_marching(
                    sigma, torch.exp(rgb_h), zvals, inference_only=inference_only, bkg_color=bkg_color
                )['rgb']
            # for unit loss
            if not inference_only:
                output_fine['unit_exp'] = self.point_constraint(
                    self.fine_exp_r_mlps, self.fine_exp_g_mlps, self.fine_exp_b_mlps
                )

            # handle progress
            output['fine'] = self.output_get_progress(output_fine, get_progress)

        # adjust two stage output
        output = self.adjust_coarse_fine_output(output, inference_only)

        return output

    def point_constraint(self, l_r, l_g, l_b):
        """Use zeros rgb_h value to constrain the output value"""
        device = next(self.coarse_exp_b_mlps.parameters()).device
        zeros_x = torch.zeros([1, 3], device=device)
        ones_x = torch.ones([
            1,
        ], device=device)
        rgb_l = self.forward_exp_mlps(l_r, l_g, l_b, zeros_x, ones_x)  # log(1) = 0.0

        return rgb_l
