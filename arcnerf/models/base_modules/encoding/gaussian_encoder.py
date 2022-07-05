# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn


class GaussianEmbedder(nn.Module):
    """
        Gaussian Embedder module.
        Transfer zvals with rays into mean/cov and turn into higher dimensions.
        ref: https://github.com/google/mipnerf/blob/main/internal/mip.py
    """

    def __init__(self, gaussian_fn='cone', ipe_embed_freq=12, *args, **kwargs):
        """
        Args:
            gaussian_fn: 'cone' or 'cylinder' to model the interval
            ipe_embed_freq: freq for integrated positional encoding
        """
        super(GaussianEmbedder, self).__init__()
        self.gaussian_fn = gaussian_fn
        self.ipe_embed_freq = ipe_embed_freq
        assert self.ipe_embed_freq > 0, 'ipe_embed_freq should be non-negative'

    def get_output_dim(self):
        """Get output dim"""
        return self.ipe_embed_freq * 6

    def forward(self, zvals: torch.Tensor, rays_o: torch.Tensor, rays_d: torch.Tensor, rays_r: torch.Tensor):
        """Get embed from gaussian distribution.

        Args:
            zvals: torch.tensor (B, N+1) sample zvals for each intervals.
            rays_o: torch.tensor (B, 3) rays origin
            rays_d: torch.tensor (B, 3) rays direction
            rays_r: torch.tensor (B, 1) radius

        Returns:
            pts_embed: embedded of mean xyz in all intervals, (B, N, 6F)
        """
        batch_size, n_interval = zvals.shape[0], zvals.shape[1] - 1
        # get conical frustum
        means, covs = self.get_conical_frustum(zvals, rays_o, rays_d, rays_r)  # (B, N, 3) * 2
        means = means.view(-1, 3)  # (BN, 3)
        covs = covs.view(-1, 3)  # (BN, 3)

        # integrated_pos_enc
        pts_embed, _ = self.integrated_pos_enc(means, covs)  # (BN, 6F)
        pts_embed = pts_embed.view(batch_size, n_interval, -1)  # (B, N, 6F)

        return pts_embed

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

    def integrated_pos_enc(self, means, covs):
        """Get positional encoding from means/cov representation

        Args:
            means: means of the ray (B, 3)
            covs: covariances of the ray (B, 3)

        Returns:
            embed_mean_out: embedded of mean xyz, (B, 6F)
            embed_cov_out: embedded of cov (B, 6F)
        """
        scales = [2**i for i in range(0, self.ipe_embed_freq)]
        embed_mean = []
        embed_cov = []
        for scale in scales:
            embed_mean.append(means * scale)
            embed_cov.append(covs * scale**2)
        embed_mean = torch.cat(embed_mean, dim=-1)  # (B, 3F)
        embed_cov = torch.cat(embed_cov, dim=-1)  # (B, 3F)
        embed_mean = torch.cat([embed_mean, embed_mean + 0.5 * np.pi], dim=-1)  # (B, 6F)
        embed_cov = torch.cat([embed_cov, embed_cov], dim=-1)  # (B, 6F)

        def safe_trig(x, fn, t=100 * np.pi):
            return fn(torch.where(torch.abs(x) < t, x, x % t))

        embed_mean_out = torch.exp(-0.5 * embed_cov) * safe_trig(embed_mean, torch.sin)  # (B, 6F)
        embed_cov_out = torch.clamp_min(
            0.5 * (1 - torch.exp(-2.0 * embed_cov) * safe_trig(2.0 * embed_mean, torch.cos)) - embed_mean_out**2, 0.0
        )  # (B, 6F)

        return embed_mean_out, embed_cov_out
