# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BaseGeoNet(nn.Module):
    """Basic Geometry network.
     Input xyz coord, get geometry value like density, sdf, occupancy, etc
     """

    def __init__(self):
        super(BaseGeoNet, self).__init__()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: torch.tensor (B, input_ch)

        Returns:
            out: tensor in shape (B, 1) for geometric value(sdf, sigma, occ).
            out_feat: tensor in shape (B, W_feat) if W_feat > 0. None if W_feat <= 0
        """
        raise NotImplementedError('You must implement this function')

    def forward_geo_value(self, x: torch.Tensor):
        """Only get geometry value like sigma/sdf/occ. In shape (B,) """
        return self.forward(x)[0][:, 0]

    def forward_with_grad(self, x: torch.Tensor):
        """Get the grad of geo_value wrt input x. It could be the normal on surface"""
        with torch.enable_grad():
            x = x.requires_grad_(True)
            geo_value, h = self.forward(x)
            grad = torch.autograd.grad(
                outputs=geo_value,
                inputs=x,
                grad_outputs=torch.ones_like(geo_value, requires_grad=False, device=x.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

        return geo_value, h, grad

    def pretrain_siren(self, n_iter=5000, lr=1e-4, thres=0.01, n_pts=5000):
        """Pretrain the siren params. Implement if you need to use it since it is called outside"""
        return


class BaseRadianceNet(nn.Module):
    """Basic Radiance network.
    Input view direction(optional: xyz dir, feature, norm, etc), get rgb color.
    """

    def __init__(self):
        super(BaseRadianceNet, self).__init__()

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor, normals: torch.Tensor, geo_feat: torch.Tensor):
        """
        Args:
            any of x/view_dir/normals/geo_feat are optional, based on mode
            x: torch.tensor (B, input_ch_pts)
            view_dirs: (B, input_ch_view)
            normals: (B, 3)
            geo_feat: (B, W_feat_in)

        Returns:
            out: tensor in shape (B, 3) for radiance value(rgb).
        """
        raise NotImplementedError('You must implement this function')
