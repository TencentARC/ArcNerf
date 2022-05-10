# img_loss
Loss for comparing image. Used for view systhesis.
## ImgCFLoss / ImgCFL1Loss
For two stage model like nerf(coarse+fine). MSE/L1 Loss for rgb values.
- gt
  - 'img': rgb value in float (0~1). `(B, N_rays, 3)`
- Output:
  - 'rgb_coarse': coarse rgb from ray marching. `(B, N_rays, 3)`
  - 'rgb_fine': fine rgb from ray marching. `(B, N_rays, 3)`

## ImgLoss / ImgL1Loss
Single pass model. MSE/L1 Loss for rgb values.
- gt
  - 'img': rgb value in float (0~1). `(B, N_rays, 3)`
- Output:
  - 'rgb': rgb from ray marching. `(B, N_rays, 3)`

## ImgCFMaskLoss / ImgCFMaskL1Loss
For two stage model like nerf(coarse+fine) with mask. MSE/L1 Loss for rgb values.
Mask is adjust on each sample instead of meaning all pixels together.
- gt
  - 'img': rgb value in float (0~1). `(B, N_rays, 3)`
  - 'mask': mask value in float (0~1). `(B, N_rays)`
- Output:
  - 'rgb_coarse': coarse rgb from ray marching. `(B, N_rays, 3)`
  - 'rgb_fine': fine rgb from ray marching. `(B, N_rays, 3)`

## ImgMaskLoss / ImgMaskL1Loss
Single pass model with mask. MSE/L1 Loss for rgb values.
Mask is adjust on each sample instead of meaning all pixels together.
- gt
  - 'img': rgb value in float (0~1). `(B, N_rays, 3)`
  - 'mask': mask value in float (0~1). `(B, N_rays)`
- Output:
  - 'rgb': rgb from ray marching. `(B, N_rays, 3)`


# mask_loss
Loss for comparing mask. Used for view systhesis and object area prediction.
## MaskCFLoss / MaskCFL1Loss / MaskCFBCELoss
For two stage model like nerf(coarse+fine). MSE/L1/BCE Loss for mask values.
- gt
  - 'mask': mask value in float (0~1). `(B, N_rays)`
- Output:
  - 'mask_coarse': coarse mask from ray marching. `(B, N_rays)`
  - 'mask_fine': fine mask from ray marching. `(B, N_rays)`
- for `BCE` Loss, the outputs should be clipped in `(1e-3, 1.0-1e-3)`

## MaskLoss / MaskL1Loss / MaskBCELoss
Single pass model. MSE/L1/BCE Loss for mask values.
- gt
  - 'mask': mask value in float (0~1). `(B, N_rays)`
- Output:
  - 'mask': mask from ray marching. `(B, N_rays)`
  - for `BCE` Loss, the output should be clipped in `(1e-3, 1.0-1e-3)`


# geo_loss
Loss for geometric regularization
## EikonalLoss
Regularize normal each ray that each normal has norma 1, MSE Loss
- Output:
  - 'normal': normal from implicit function. `(B, N_rays, 3)`
## EikonalPTLoss
Regularize normal for all pts that each normal has norma 1, MSE Loss
- Output:
  - 'normal_pts': normal from implicit function. `(B, N_rays, N_pts, 3)`
## EikonalMaskLoss
Regularize normal each ray that each normal has norma 1, MSE Loss, with mask
- Output:
  - 'normal': normal from implicit function. `(B, N_rays, 3)`
- gt
  - 'mask': mask value in float (0~1). `(B, N_rays)`
## EikonalPTMaskLoss
Regularize normal for all pts that each normal has norma 1, MSE Loss, with mask
- Output:
  - 'normal_pts': normal from implicit function. `(B, N_rays, N_pts, 3)`
- gt
  - 'mask': mask value in float (0~1). `(B, N_rays)`
