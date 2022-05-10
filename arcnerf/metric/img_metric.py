# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from torchgeometry.losses.ssim import SSIM as _SSIM
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

        # avg mse first in case inf psnr
        mse = self.mse(output[self.key], gt)  # (B, N_rays, 3)
        if self.use_mask:
            mse = mean_tensor_by_mask(mse.mean(-1), mask)
        else:
            mse = mse.mean()

        psnr = -10.0 * torch.log10(mse)

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


@METRIC_REGISTRY.register()
class SSIM(nn.Module):
    """SSIM for image and gt
    call torchgeometry: https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/ssim.html
    """

    def __init__(self, cfgs=None):
        super(SSIM, self).__init__()
        self.ssim = _SSIM(window_size=3, reduction='none')
        self.key = 'rgb'
        self.use_mask = False

    def forward(self, data, output):
        """
        Args:
            output['rgb']: (B, N_rays, 3). img output
            data['img']: (B, N_rays, 3)
            data['mask']: (B, N_rays), only if used mask
            data['H']: (B,) image height
            data['W']: (B,) image width

        Returns:
            ssim: (1, ) mean ssim.
        """
        device = output[self.key].device
        gt = data['img'].to(device)
        if self.use_mask:
            mask = data['mask'].to(device)

        # reshape and transpose
        H, W = int(data['H'][0]), int(data['W'][0])
        assert H * W == gt.shape[1], 'invalid shape. HW does not match n_rays...'
        gt_reshape = gt.view(-1, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
        output_reshape = output[self.key].view(-1, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
        if self.use_mask:
            mask_reshape = mask.view(-1, H, W)  # (B, H, W)

        # get ssim and mean
        ssim = self.ssim(output_reshape, gt_reshape).permute(0, 2, 3, 1)  # (B, H, W, 3)

        if self.use_mask:
            ssim = mean_tensor_by_mask(ssim.mean(-1), mask_reshape)
        else:
            ssim = ssim.mean()

        return ssim


@METRIC_REGISTRY.register()
class SSIMCoarse(SSIM):
    """SSIM for coarse image output and gt"""

    def __init__(self, cfgs=None):
        super(SSIMCoarse, self).__init__()
        self.key = 'rgb_coarse'


@METRIC_REGISTRY.register()
class SSIMFine(SSIM):
    """SSIM for fine image output and gt"""

    def __init__(self, cfgs=None):
        super(SSIMFine, self).__init__()
        self.key = 'rgb_fine'


@METRIC_REGISTRY.register()
class MaskSSIM(SSIM):
    """SSIM for image and gt with mask"""

    def __init__(self, cfgs=None):
        super(MaskSSIM, self).__init__(cfgs)
        self.use_mask = True


@METRIC_REGISTRY.register()
class MaskSSIMCoarse(SSIMCoarse):
    """SSIM for coarse image and gt with mask"""

    def __init__(self, cfgs=None):
        super(MaskSSIMCoarse, self).__init__(cfgs)
        self.use_mask = True


@METRIC_REGISTRY.register()
class MaskSSIMFine(SSIMFine):
    """SSIM for fine image and gt with mask"""

    def __init__(self, cfgs=None):
        super(MaskSSIMFine, self).__init__(cfgs)
        self.use_mask = True
