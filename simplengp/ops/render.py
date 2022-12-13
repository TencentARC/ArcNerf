# -*- coding: utf-8 -*-

import torch
try:
    import _render
except ImportError:
    raise NotImplementedError("You have not build the customized ops...run `sh scripts/install_ops.sh`...")


# -------------------------------------------------- ------------------------------------ #

class CalRgbBP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigma, radiance, rays_numsteps, dt, training_background_color, early_stop):

        # output tensor
        rays_numsteps = rays_numsteps.detach()
        n_rays_per_batch = rays_numsteps.shape[0]
        rgb_output = torch.zeros((n_rays_per_batch, 3), dtype=sigma.dtype, device=sigma.device)
        alpha_output = torch.zeros((n_rays_per_batch, 1), dtype=sigma.dtype, device=sigma.device)

        _render.calc_rgb_forward(
            sigma, radiance, rays_numsteps, dt, training_background_color, rgb_output, alpha_output, early_stop
        )

        ctx.save_for_backward(sigma, radiance, rays_numsteps, dt, rgb_output)
        ctx.extro = early_stop

        return rgb_output, alpha_output

    @staticmethod
    def backward(ctx, grad_rgb, grad_alpha):

        sigma, radiance, rays_numsteps, dt, rgb_output = ctx.saved_tensors
        early_stop = ctx.extro

        # grad output tensor
        num_elements = sigma.shape[0]
        grad_sigma = torch.zeros((num_elements,), dtype=sigma.dtype, device=sigma.device)
        grad_radiance = torch.zeros((num_elements, 3), dtype=sigma.dtype, device=sigma.device)

        _render.calc_rgb_backward(
            sigma, radiance, rays_numsteps, dt, rgb_output,
            grad_rgb, grad_sigma, grad_radiance, early_stop
        )

        return grad_sigma, grad_radiance, None, None, None, None


def calc_rgb_bp(sigma, radiance, rays_numsteps, dt, training_background_color, early_stop):
    """Calculate rgb with backward propagation"""
    return CalRgbBP.apply(sigma, radiance, rays_numsteps, dt, training_background_color, early_stop)

# -------------------------------------------------- ------------------------------------ #


class CalRgbNoBP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigma, radiance, rays_numsteps, dt, bg_color, early_stop):
        rays_numsteps = rays_numsteps.detach()

        # output tensor
        n_rays_per_batch = rays_numsteps.shape[0]
        rgb_output = torch.zeros((n_rays_per_batch, 3), dtype=sigma.dtype, device=sigma.device)
        alpha_output = torch.zeros((n_rays_per_batch, 1), dtype=sigma.dtype, device=sigma.device)

        _render.calc_rgb_inference(sigma, radiance, rays_numsteps, dt, bg_color, rgb_output, alpha_output, early_stop)

        return rgb_output.detach(), alpha_output.detach()


@torch.no_grad()
def calc_rgb_nobp(sigma, radiance, rays_numsteps, dt, bg_color, early_stop):
    """Calculate the rgb in forward only way"""
    return CalRgbNoBP.apply(sigma, radiance, rays_numsteps, dt, bg_color, early_stop)


# -------------------------------------------------- ------------------------------------ #

class FillInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigma, radiance, dt, numsteps_in, _sigma, _radiance, _dt):

        _render.fill_input_forward(sigma, radiance, dt, numsteps_in, _sigma, _radiance, _dt)

        ctx.save_for_backward(sigma, radiance, dt, numsteps_in)

        return _sigma, _radiance, _dt

    @staticmethod
    def backward(ctx, _grad_sigma, _grad_radiance, _grad_dt):

        sigma, radiance, dt, numsteps_in = ctx.saved_tensors

        # grad output for sigma/radiance
        grad_sigma = torch.zeros_like(sigma, dtype=_grad_sigma.dtype, device=_grad_sigma.device)
        grad_radiance = torch.zeros_like(radiance, dtype=_grad_sigma.dtype, device=_grad_sigma.device)
        grad_dt = torch.zeros_like(dt, dtype=_grad_sigma.dtype, device=_grad_sigma.device)

        _render.fill_input_backward(
            grad_sigma, grad_radiance, grad_dt, numsteps_in, _grad_sigma, _grad_radiance, _grad_dt
        )

        return grad_sigma, grad_radiance, grad_dt, None, None, None, None


def fill_ray_marching_inputs(sigma, radiance, dt, numsteps_in, _sigma, _radiance, _dt):
    """This function is used for make the flatten sigma/radiance in (n_pts,...) shape into (n_rays, n_pts, ...)
        So that it could use the original torch rendering function in nerf
    """
    return FillInput.apply(sigma, radiance, dt, numsteps_in, _sigma, _radiance, _dt)
