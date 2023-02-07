# img_loss
Loss for comparing image. Used for view synthesis.
## ImgLoss
- keys: list of key to calculate, used sum of them. By default `['rgb']`.
`['rgb_coarse/fine']` for two stage model.
- loss_type: select loss type such as 'MSE'/'L1'/'huber'. By default MSE
- internal_weights: If set, will multiply factors to each weight. By default None.
- use_mask: use mask for average calculation. By default False.
- do_mean: calculate the mean of loss. By default True.

Required input/output:
- gt
  - 'img': rgb value in float (0~1). `(B, N_rays, 3)`
- Output:
  - 'rgb/rgb_coarse/rgb_fine': one or several rgb values by keys. `(B, N_rays, 3)`
  - 'mask': if use_mask. `(B, N_rays)`

## FixValueLoss
The loss of any tensor to a fix_value
- keys: list of key to calculate
- fix_value: the fixed floating target
- internal_weights: If set, will multiply factors to each weight. By default None.

Required input/output:
- Output: Any key specified in `keys` with tensor as value.

------------------------------------------------------------------------
# mask_loss
Loss for comparing mask. Used for view synthesis and object area prediction.
## MaskLoss
- keys: list of key to calculate, used sum of them. By default `['mask']`.
`['mask_coarse/fine']` for two stage model.
- loss_type: select loss type such as 'MSE'/'L1'/'BCE'. By default MSE.
  - for 'BCE' loss it will clip the output to avoid error.
- do_mean: calculate the mean of loss. By default True.

Required input/output:
- gt
  - 'mask': mask value in float (0~1). `(B, N_rays)`
- Output:
  - 'mask/mask_coarse/mask_fine': one or several mask values by keys. `(B, N_rays)`

------------------------------------------------------------------------
# geo_loss
Loss for geometric regularization
## EikonalLoss
Regularize normal each ray/pts that each normal has normal 1.
- key: Single key to calculate. By default `['normal']`.
`['normal_pts']` for all sample points.
- loss_type: select loss type such as 'MSE'/'L1'. By default MSE.
- do_mean: calculate the mean of loss. By default True.

Required output:
- Output:  (select by key)
  - 'normal': normal from implicit function. `(B, N_rays, 3)`
  - 'normal_pts': normal points from implicit function. `(B, N_rays, N_pts, 3)`
## RegMaskLoss
Regularize mask each ray that each value is 0 or 1, opacity clear.
- keys: keys to calculate. By default `['mask']`.  `['mask_coarse/fine']` for two stage model like nerf.
- do_mean: calculate the mean of loss. By default True.

Required output:
- Output:  (select by key)
  - 'mask': mask output. `(B, N_rays)`
## RegWeightsLoss
Regularize weights each ray pts that each value is 0 or 1, opacity clear.
- keys: keys to calculate. By default `['weights']`.  `['weights_coarse/fine']` for two stage model like nerf.
  - real key in output is starting with `progress_`. (You must set `debug.get_progress=True`)
- do_mean: calculate the mean of loss. By default True.

Required output:
- Output:  (select by key)
  - 'progress_weights': ray marching progress output. `(B, N_rays, N_pts)`
