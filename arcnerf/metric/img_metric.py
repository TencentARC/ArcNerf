# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from common.utils.registry import METRIC_REGISTRY
from common.utils.torch_utils import mean_tensor_by_mask


@METRIC_REGISTRY.register()
class PSNR(nn.Module):
    """PSNR for image and gt"""

    def __init__(self, cfgs=None):
        super(PSNR, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.key = 'rgb'
        self.use_mask = False

    def forward(self, data, output):
        """
        Args:
            output['rgb']: (B, N_rays, 3). img output
            data['img']: (B, N_rays, 3)
            data['mask']: (B, N_rays), only if used mask

        Returns:
            psnr: (1, ) mean psnr.
        """
        device = output[self.key].device
        gt = data['img'].to(device)
        if self.use_mask:
            mask = data['mask'].to(device)

        psnr = -10.0 * torch.log10(self.mse(output[self.key], gt))  # (B, N_rays, 3)
        if self.use_mask:
            psnr = mean_tensor_by_mask(psnr.mean(-1), mask)
        else:
            psnr = psnr.mean()

        return psnr


@METRIC_REGISTRY.register()
class PSNRCoarse(PSNR):
    """PSNR for coarse image output and gt"""

    def __init__(self, cfgs=None):
        super(PSNRCoarse, self).__init__(cfgs)
        self.key = 'rgb_coarse'


@METRIC_REGISTRY.register()
class PSNRFine(PSNR):
    """PSNR for fine image output and gt"""

    def __init__(self, cfgs=None):
        super(PSNRFine, self).__init__(cfgs)
        self.key = 'rgb_fine'


@METRIC_REGISTRY.register()
class MaskPSNR(PSNR):
    """PSNR for image and gt with mask"""

    def __init__(self, cfgs=None):
        super(MaskPSNR, self).__init__(cfgs)
        self.use_mask = True


@METRIC_REGISTRY.register()
class MaskPSNRCoarse(PSNRCoarse):
    """PSNR for coarse image and gt with mask"""

    def __init__(self, cfgs=None):
        super(MaskPSNRCoarse, self).__init__(cfgs)
        self.use_mask = True


@METRIC_REGISTRY.register()
class MaskPSNRFine(PSNRFine):
    """PSNR for fine image and gt with mask"""

    def __init__(self, cfgs=None):
        super(MaskPSNRFine, self).__init__(cfgs)
        self.use_mask = True
