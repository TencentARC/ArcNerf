# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
class PSNR(nn.Module):
    """PSNR for image and gt

    PSNR = -10*log10(MSE) /  MSE = 10 ** (-PSRN/10)
    PSNR:    10,   20,    23,    25,     28,    30,     32,     35
    MSE:    0.1, 0.01, 0.005, 0.003, 0.0016, 0.001, 0.0006, 0.0003
    """

    def __init__(self, cfgs=None):
        """
        Args:
            cfgs: a obj with following attributes:
                key: key used for loss sum. By default 'rgb'.
                      'rgb_coarse'/'rgb_fine' for two stage network
                use_mask: use mask for average calculation. By default False.
        """
        super(PSNR, self).__init__()
        self.key = get_value_from_cfgs_field(cfgs, 'key', 'rgb')
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, data, output):
        """
        Args:
            output['rgb'/'rgb_coarse'/'rgb_fine']: (B, N_rays, 3). img output based on key
            data['img']: (B, N_rays, 3)

        Returns:
            psnr: (1, ) mean psnr.
        """
        device = output[self.key].device
        gt = data['img'].to(device)

        # avg mse first in case inf psnr
        mse = self.mse(output[self.key], gt)  # (B, N_rays, 3)
        mse = mse.mean()

        psnr = -10.0 * torch.log10(mse)

        return psnr
