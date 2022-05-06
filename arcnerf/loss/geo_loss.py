# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from common.utils.registry import LOSS_REGISTRY
from common.utils.torch_utils import mean_tensor_by_mask


@LOSS_REGISTRY.register()
class EikonalLoss(nn.Module):
    """Eikonal MSE Loss for normal map, regularize the normal has norm 1"""

    def __init__(self, cfgs=None):
        super(EikonalLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')
        self.key = 'normal'
        self.use_mask = False
        self.pts = False

    def forward(self, data, output):
        """
        Args:
            output['normal'/'normal_pts']: (B, N_rays, (N_pts), 3). normal output
            data['mask']: (B, N_rays), only if used mask

        Returns:
            Eikonal: (1, ) mean Eikonal loss.
        """
        dtype = output[self.key].dtype
        device = output[self.key].device
        out = output[self.key]
        norm = torch.norm(out, dim=-1)  # (B, n_rays, (n_pts))
        norm_ones = torch.ones_like(norm, dtype=dtype).to(device)
        if self.use_mask:
            mask = data['mask'].to(device)  # (B, n_rays)

        loss = self.loss(norm, norm_ones)
        if self.use_mask:
            if self.pts:  # expand for pts-dim
                mask = torch.repeat_interleave(mask.unsqueeze(-1), loss.shape[-1], -1)
            loss = mean_tensor_by_mask(loss, mask)
        else:
            loss = loss.mean()

        return loss


@LOSS_REGISTRY.register()
class EikonalPTLoss(EikonalLoss):
    """Eikonal MSE Loss for normal pts"""

    def __init__(self, cfgs=None):
        super(EikonalPTLoss, self).__init__(cfgs)
        self.key = 'normal_pts'
        self.pts = True


@LOSS_REGISTRY.register()
class EikonalMaskLoss(EikonalLoss):
    """Eikonal MSE Loss for normal map, with mask"""

    def __init__(self, cfgs=None):
        super(EikonalMaskLoss, self).__init__(cfgs)
        self.use_mask = True


@LOSS_REGISTRY.register()
class EikonalPTMaskLoss(EikonalPTLoss):
    """Eikonal MSE Loss for normal pts, with mask"""

    def __init__(self, cfgs=None):
        super(EikonalPTMaskLoss, self).__init__(cfgs)
        self.use_mask = True
