# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from . import ENCODER_REGISTRY


class Gaussian(nn.Module):
    """
        Gaussian module.
        Transfer zvals with rays into mean/cov
        ref: https://github.com/google/mipnerf/blob/main/internal/mip.py
    """

    def __init__(self, gaussian_fn='cone'):
        """
        Args:
            gaussian_fn: 'cone' or 'cylinder' to model the interval
        """
        super(Gaussian, self).__init__()
        self.gaussian_fn = gaussian_fn

    def forward(self, zvals: torch.Tensor, rays_o: torch.Tensor, rays_d: torch.Tensor, rays_r: torch.Tensor):
        """Get embed from gaussian distribution.

        Args:
            zvals: torch.tensor (B, N+1) sample zvals for each intervals.
            rays_o: torch.tensor (B, 3) rays origin
            rays_d: torch.tensor (B, 3) rays direction
            rays_r: torch.tensor (B, 1) radius

        Returns:
            mean_cov: gaussian representation of the interval (B, N, 3*2), first 3-mean, second 3-cov
        """
        means, covs = self.get_conical_frustum(zvals, rays_o, rays_d, rays_r)  # (B, N, 3) * 2
        mean_cov = torch.cat([means, covs], dim=-1)  # (B, N, 3*2)

        return mean_cov

    def get_conical_frustum(self, zvals, rays_o, rays_d, rays_r):
        """Get the mean/cov representation of the conical frustum

        Args:
            zvals: torch.tensor (B, N+1) sample zvals for each intervals.
            rays_o: torch.tensor (B, 3) rays origin
            rays_d: torch.tensor (B, 3) rays direction
            rays_r: torch.tensor (B, 1) radius

        Returns:
            means: means of the ray (B, N, 3)
            covs: covariances of the ray (B, N, 3)
        """
        t_start = zvals[:, :-1]
        t_end = zvals[:, 1:]
        if self.gaussian_fn == 'cone':
            gaussian_fn = self.conical_frustum_to_gaussian
        elif self.gaussian_fn == 'cylinder':
            gaussian_fn = self.cylinder_to_gaussian
        else:
            raise NotImplementedError('Invalid gaussian function {}'.format(self.gaussian_fn))

        means, covs = gaussian_fn(rays_d, t_start, t_end, rays_r)  # (B, N, 3) * 2
        means = means + rays_o.unsqueeze(1)  # (B, N, 3)

        return means, covs

    def conical_frustum_to_gaussian(self, rays_d, t_start, t_end, rays_r):
        """Turn conical frustum into gaussian representation
        Sec 3.1 in paper

        Args:
            rays_d: torch.tensor (B, 3) rays direction
            t_start: (B, N) start zvals for each interv
            t_end: (B, N) end zvals for each interval
            rays_r: torch.tensor (B, 1) basic radius

        Returns:
            means: means of the ray (B, N, 3)
            covs: covariances of the ray (B, N, 3)
        """
        mu = (t_start + t_end) / 2.0  # (B, N)
        hw = (t_end - t_start) / 2.0  # (B, N)
        common_term = 3.0 * mu**2 + hw**2  # (B, N)
        t_mean = mu + (2.0 * mu * hw**2) / common_term  # (B, N)
        t_var = (hw**2) / 3.0 - (4.0 / 15.0) * ((hw**4 * (12.0 * mu**2 - hw**2)) / common_term**2)  # (B, N)
        r_var = rays_r**2 * ((mu**2) / 4.0 + (5.0 / 12.0) * hw**2 - (4.0 / 15.0) * (hw**4) / common_term)  # (B, N)
        mean, covs = self.lift_gaussian(rays_d, t_mean, t_var, r_var)

        return mean, covs

    def cylinder_to_gaussian(self, rays_d, t_start, t_end, rays_r):
        """Turn cylinder frustum into gaussian representation

        Args:
            rays_d: torch.tensor (B, 3) rays direction
            t_start: (B, N) start zvals for each interv
            t_end: (B, N) end zvals for each interval
            rays_r: torch.tensor (B, 1) radius

        Returns:
            means: means of the ray (B, N, 3)
            covs: covariances of the ray (B, N, 3)
        """
        t_mean = (t_start + t_end) / 2.0  # (B, N)
        t_var = (t_end - t_start)**2 / 12.0  # (B, N)
        r_var = rays_r**2 / 4.0  # (B, N)
        mean, covs = self.lift_gaussian(rays_d, t_mean, t_var, r_var)

        return mean, covs

    @staticmethod
    def lift_gaussian(rays_d, t_mean, t_var, r_var):
        """Lift mu/t to rays gaussian mean/var

        Args:
            rays_d: direction (B, 3)
            t_mean: mean (B, N) of each interval along ray
            t_var: variance (B, N) of each interval along ray
            r_var: variance (B, N) of each interval perpendicular to ray

        Returns:
            means: means of the ray (B, N, 3)
            covs: covariances of the ray (B, N, 3)
        """
        mean = rays_d.unsqueeze(1) * t_mean.unsqueeze(-1)  # (B, N, 3)
        d_mag_sq = torch.clamp_min(torch.sum(rays_d**2, dim=-1, keepdim=True), 1e-10)  # (B, 3)
        d_outer_diag = rays_d**2  # (B, 3)
        null_outer_diag = 1 - d_outer_diag / d_mag_sq  # (B, 3)
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]  # (B, N, 3)
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]  # (B, N, 3)
        cov_diag = t_cov_diag + xy_cov_diag  # (B, N, 3)

        return mean, cov_diag


@ENCODER_REGISTRY.register()
class GaussianEmbedder(nn.Module):
    """
        GaussianEmbedder module. Embed gaussian representation(B, G*2) into higher dimensions.
        For example, x = exp(-0.5*2**2N*cov) * sin(2**N * mean) for N in range(0, 10)
        ref: https://github.com/ventusff/neurecon/blob/main/models/base.py
    """

    def __init__(
        self,
        input_dim,
        n_freqs,
        log_sampling=True,
        include_input=True,
        periodic_fns=(torch.sin, torch.cos),
        *args,
        **kwargs
    ):
        """
        Args:
            input_dim: dimension of input to be embedded. For mean and cov each.
            n_freqs: number of frequency bands. If 0, will not encode the inputs.
            log_sampling: if True, use log factor exp(-0.5 * 2**2N * cov) * sin(2**N * x).
                         Else use scale factor exp(-0.5 * N**2 * cov) sin(N * x). By default is True
            include_input: if True, raw input is included in the embedding. Appear at beginning. By default is True
            periodic_fns: a list of periodic functions used to embed input. By default is (sin, cos)

        Returns:
            Embedded inputs with shape:
                (inputs_dim * len(periodic_fns) * N_freq + include_input * inputs_dim)
            For example, inputs_dim = 3, using (sin, cos) encoding, N_freq = 10, include_input, will results at
                3 * 2 * 10 + 3 = 63 output shape.
        """
        super(GaussianEmbedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        # get output dim
        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim
        self.out_dim += self.input_dim * n_freqs * len(self.periodic_fns)

        if n_freqs == 0 and include_input:  # inputs only
            self.freq_bands = []
        else:
            if log_sampling:
                self.freq_bands = 2.**torch.linspace(0., n_freqs - 1, n_freqs)
            else:
                self.freq_bands = torch.linspace(2.**0., 2.**(n_freqs - 1), n_freqs)

    def get_output_dim(self):
        """Get output dim"""
        return self.out_dim

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: tensor of shape [B, input_dim*2], mean/cov combination

        Returns:
            embed_x: tensor of shape [B, out_dim]
        """
        assert (x.shape[-1] == self.input_dim * 2), 'Input shape should be (B, 2*{})'.format(self.input_dim)

        means, covs = x[:, :self.input_dim], x[:, self.input_dim:]

        embed_x = []
        if self.include_input:
            embed_x.append(means)

        for freq in self.freq_bands:
            for fn in self.periodic_fns:
                embed_x.append(torch.exp(-0.5 * freq**2 * covs) * fn(means * freq))

        if len(embed_x) > 1:
            embed_x = torch.cat(embed_x, dim=-1)
        else:
            embed_x = embed_x[0]

        return embed_x
