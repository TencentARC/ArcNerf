# img_metric
Metric for evaluating image quality
## PSNR
PSNR between images, calculate from mean mse loss.
- key: used for selecting rgb from output. By default rgb. Can be 'rgb_fine/coarse' if needed.
- use_mask: eval on mask area only. False here.
## MaskPSNR
Child of PSNR.
- use_mask: Set to True

## SSIM/MaskSSIM
Same as PSNR but use different metric.
