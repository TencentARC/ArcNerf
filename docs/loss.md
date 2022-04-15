# img_loss
Loss for comparing image. Used for view systhesis.
## ImgCFLoss
For two stage model like nerf(coarse+fine). MSE Loss for rgb values.
- gt
  - 'img': rgb value in float (0~1). `(B, N_rays, 3)`
- Output:
  - 'rgb_coarse': coarse rgb from ray marching. `(B, N_rays, 3)`
  - 'rgb_fine': fine rgb from ray marching. `(B, N_rays, 3)`

## ImgLoss
Single pass model. MSE Loss for rgb values.
- gt
  - 'img': rgb value in float (0~1). `(B, N_rays, 3)`
- Output:
  - 'rgb': rgb from ray marching. `(B, N_rays, 3)`


## ImgCFMaskLoss
For two stage model like nerf(coarse+fine) with mask. MSE Loss for rgb values.
Mask is adjust on each sample instead of meaning all pixels together.
- gt
  - 'img': rgb value in float (0~1). `(B, N_rays, 3)`
  - 'mask': mask value in float (0~1). `(B, N_rays)`
- Output:
  - 'rgb_coarse': coarse rgb from ray marching. `(B, N_rays, 3)`
  - 'rgb_fine': fine rgb from ray marching. `(B, N_rays, 3)`

## ImgMaskLoss
Single pass model with mask. MSE Loss for rgb values.
Mask is adjust on each sample instead of meaning all pixels together.
- gt
  - 'img': rgb value in float (0~1). `(B, N_rays, 3)`
  - 'mask': mask value in float (0~1). `(B, N_rays)`
- Output:
  - 'rgb': rgb from ray marching. `(B, N_rays, 3)`
