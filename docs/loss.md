# img_loss
Loss for comparing image. Used for view systhesis.
## ImgLoss
- keys: list of key to calculate, used sum of them. By default `['rgb']`.
`['rgb_coarse/fine']` for two stage model.
- loss_type: select loss type such as 'MSE'/'L1'. By default MSE
- use_mask: use mask for average calculation. By default False.
- do_mean: calculate the mean of loss. By default True.

Required input/output:
- gt
  - 'img': rgb value in float (0~1). `(B, N_rays, 3)`
- Output:
  - 'rgb/rgb_coarse/rgb_fine': one or several rgb values by keys. `(B, N_rays, 3)`
  - 'mask': if use_mask. `(B, N_rays)`

------------------------------------------------------------------------
# mask_loss
Loss for comparing mask. Used for view systhesis and object area prediction.
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
