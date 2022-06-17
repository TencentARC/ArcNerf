# img_metric
Metric for evaluating image quality

## PSNR
PSNR between images, calculate from mean mse loss.
- key: used for selecting rgb from output. By default rgb. Can be 'rgb_fine/coarse' if needed.
- use_mask: eval on mask area only. False here.

## MaskPSNR
Child of PSNR. Only eval masked area.
- use_mask: Set to True

## SSIM/MaskSSIM
Same as PSNR but use different metric.

------------------------------------------------------------------------
# train_metric
You can specify the train_metric to get the current training eval metric. But only use one
metric in train mode.
