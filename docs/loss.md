# img_loss
## ImgLoss
For two stage model like nerf(coarse+fine). MSE Loss for rgb values.
- gt
  - 'img': rgb value in float (0~1). `(B, N_rays, 3)`
- Output:
  - 'rgb_coarse': coarse rgb from ray marching. `(B, N_rays, 3)`
  - 'rgb_fine': fine rgb from ray marching. `(B, N_rays, 3)`

## ImgCFLoss
Single pass model. MSE Loss for rgb values.
- gt
  - 'img': rgb value in float (0~1). `(B, N_rays, 3)`
- Output:
  - 'rgb': rgb from ray marching. `(B, N_rays, 3)`
