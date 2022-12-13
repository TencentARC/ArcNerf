# Notice

Many issues that are not mentioned in detail in the paper affect the performance. 

Including `learning_rate_schedule`/`batch size`/`sample method`/`precrop`/`noise std`/`perturb`/`mask loss`

For llff(forward facing), also needs to consider `ndc`/`inverse_depth`

------------------------------------------------------------------------
# Mode
Two modes are support for getting the rays for training:
  - mode: `['full', 'random']`, full takes all rays, random is sample with replacement
    - 'full' mode takes all the rays for training. It will shuffle every time all rays have been processed
    - 'random': random sample rays in batch randomly with replacement. Some rays may not be sampled in this mode.
  - cross_view: used in both mode. If True, each sample takes rays from different image. Else on in one image.

------------------------------------------------------------------------
# Results
For all the experiments, we run for 300k epoch, and record results every 50k epoch.

All use `adam` optimizer with `ExponentialLR` decay.

All dataset split follow the original implementation.

------------------------------------------------------------------------
## Lego (NeRF Synthetic dataset)
Image size 800x800. Train/Eval: 100/25 (No overlap).

In original [nerf-torch](https://github.com/yenchenlin/nerf-pytorch) implementation, after `200k` epoch (batchsize=4096),
I got:
- `PSNR = 32.3`  (in paper `32.54`), No noise_std=0.0 in official configs.

Our implementation results:

| Expname | batch_size |  lr | lr_steps | ray_sample_mode | cross_view | precrop | noise_std | perturb | maskimgloss |mask_loss| PSNR | Epoch 50k | Epoch 100k | Epoch 150k | Epoch 200k | Epoch 250k | Epoch 300k |
|:-------:|:----------:|:---:|:--------:|:---------------:|:----------:|:-------:|:---------:|:-------:|:-----------:|:-------:|:----:|:---------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|lego_full_cross                        | 4096 | 5e-4 | 500k |  full  | True  | 0.5-500 | 0.0 |   True   |   No    |   No    |      |   30.03   |    31.28   |    31.84   |   32.37    |    32.57   | **32.75**  | 
|lego_random_nocross                    | 4096 | 5e-4 | 500k | random | False | 0.5-500 | 0.0 |   True   |   No    |   No    |      |   29.69   |    30.96   |    31.65   |   32.26    |    32.44   |   32.66    |
|lego_full_nocross                      | 4096 | 5e-4 | 500k |  full  | False | 0.5-500 | 0.0 |   True   |   No    |   No    |      |   26.90   |    27.80   |    28.11   |   28.33    |    28.51   |   28.62    | 
|lego_noprecrop                         | 4096 | 5e-4 | 500k |  full  | True  |    No   | 0.0 |   True   |   No    |   No    |      |   9.48    |    9.48    |    9.48    |   9.48     |    9.48    |   9.48     | 
|lego_addnoise                          | 4096 | 5e-4 | 500k |  full  | True  | 0.5-500 | 1.0 |   True   |   No    |   No    |      |   29.84   |    31.16   |    31.84   |   32.14    |    32.46   |   32.64    | 
|lego_noperturb                         | 4096 | 5e-4 | 500k |  full  | True  | 0.5-500 | 0.0 |   False  |   No    |   No    |      |   29.95   |    31.17   |    31.56   |   32.03    |    32.17   |   32.30    | 
|lego_lrstep200k                        | 4096 | 5e-4 | 200k |  full  | True  | 0.5-500 | 1.0 |   True   |   No    |   No    |      |   29.87   |    31.31   |    31.92   |   32.26    |    32.39   |   32.50    | 
|lego_lrstep200k_b1024                  | 1024 | 5e-4 | 200k |  full  | True  | 0.5-500 | 1.0 |   True   |   No    |   No    |      |   28.18   |    29.79   |    30.56   |   31.08    |    31.34   |   31.47    | 
|lego_b1024                             | 1024 | 5e-4 | 500k |  full  | True  | 0.5-500 | 1.0 |   True   |   No    |   No    |      |   28.32   |    29.61   |    30.41   |   31.05    |    31.38   |   31.77    | 
|lego_maskimgloss_maskloss01            | 4096 | 5e-4 | 500k |  full  | True  | 0.5-500 | 0.0 |   True   |   Yes   |   0.1   |      |   9.48    |    9.48    |    9.48    |   9.48     |    9.48    |   9.48     | 
|lego_maskimgloss_maskloss1             | 4096 | 5e-4 | 500k |  full  | True  | 0.5-500 | 0.0 |   True   |   Yes   |   1.0   |      |   29.52   |    30.43   |    31.11   |   31.41    |    31.72   |   31.91    | 
|lego_maskloss01                        | 4096 | 5e-4 | 500k |  full  | True  | 0.5-500 | 0.0 |   True   |   No    |   0.1   |      |   29.83   |    31.28   |    31.82   |   32.19    |    32.40   |   32.67    | 
|lego_maskloss1                         | 4096 | 5e-4 | 500k |  full  | True  | 0.5-500 | 0.0 |   True   |   No    |   1.0   |      |   28.88   |    30.08   |    30.65   |   30.88    |    31.12   |   31.28    |

------------------------------------------------------------------------

## fern (LLFF dataset)
Image size 378x504(Downsample 1/8). Train/Eval: 17:3 (No overlap).

In orginal [nerf-torch](https://github.com/yenchenlin/nerf-pytorch) implementation, after `200k` epoch（batchsize=4096),
I got:
- `ndc` mode: `PSNR = 26.95`  (in paper `25.17`)
- `no_ndc` mode: `PSNR = 26.29`

They all use ndc with view-dirs in non-ndc space. noise_std=1.0 in implementation.

Our implementation results:

| Expname | batch_size |  lr | lr_steps | ray_sample_mode | cross_view | pts ndc  | view_dirs space | noise_std | perturb | inverse_depth | PSNR | Epoch 50k | Epoch 100k | Epoch 150k | Epoch 200k | Epoch 250k | Epoch 300k |
|:-------:|:----------:|:---:|:--------:|:---------------:|:----------:|:--------:|:---------------:|:---------:|:-------:|:-------------:|:----:|:---------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|fern_no_ndc                            | 4096 | 5e-4 | 250k |  full  | True  | False |  non-ndc  | 1.0 |  True  |     False     |      |   25.45   |    25.91   |    26.07   |   26.09    |    26.07   |   26.04    |
|fern_ndc                               | 4096 | 5e-4 | 250k |  full  | True  | True  |    ndc    | 1.0 |  True  |     False     |      |   25.97   |    25.15   |    24.88   |   24.69    |    24.60   |   24.49    |
|fern_ndc_use_nondc_rays_d              | 4096 | 5e-4 | 250k |  full  | True  | True  |  non-ndc  | 1.0 |  True  |     False     |      |   26.17   |    26.63   |    26.70   |   26.66    |    26.62   |   26.55    |
|fern_ndc_random_nocross                | 4096 | 5e-4 | 250k | random | False | True  |    ndc    | 1.0 |  True  |     False     |      |   26.07   |    25.39   |    25.29   |   24.84    |    24.76   |   24.64    |
|fern_ndc_correct_invdepth              | 4096 | 5e-4 | 250k |  full  | True  | True  |  non-ndc  | 1.0 |  True  |     True      |      |   17.26   |    16.53   |    16.14   |   15.84    |    15.67   |   15.54    | 
|fern_ndc_correct_no_noise              | 4096 | 5e-4 | 250k |  full  | True  | True  |  non-ndc  | 0.0 |  True  |     False     |      |   26.51   |    26.98   |  **27.05** |   27.04    |    26.98   |   26.95    | 
|fern_ndc_correct_no_perturb            | 4096 | 5e-4 | 250k |  full  | True  | True  |  non-ndc  | 1.0 |  False |     False     |      |   26.01   |    26.34   |    26.34   |   26.30    |    26.23   |   26.17    |
|fern_ndc_correct_lrstep500k            | 4096 | 5e-4 | 500k |  full  | True  | True  |  non-ndc  | 0.0 |  True  |     False     |      |   26.38   |    26.81   |    26.93   |   26.91    |    26.85   |   26.84    | 
|fern_ndc_correct_lrstep500k_b1024      | 1024 | 5e-4 | 500k |  full  | True  | True  |  non-ndc  | 0.0 |  True  |     False     |      |   4.79    |    4.79    |    4.79    |   4.79     |    4.79    |   4.79     | 
|fern_ndc_correct_b1024                 | 1024 | 5e-4 | 250k |  full  | True  | True  |  non-ndc  | 0.0 |  True  |     False     |      |   25.54   |    26.42   |    26.77   |   26.95    |    27.02   |   27.04    | 

------------------------------------------------------------------------
# Conclusion

- Lego (White_bkg images)
  - full rays shuffle with rays from different image (full-cross) is better than original non-batch implementation.
  - precrop is important for such white_bkg case since the nerf model is sensitive to initialization.
  - add maskloss in image/mask does not have extra benefits in synthesizing new view.
  - b=4096 is better than b=1024, lrstep=500k better than lrstep=200k
  
- Fern (Forward facing)
  - ndc is helpful. And the rays to model(view_dirs) should be in non-ndc space.
  - 200k epoch is enough for that, possibly because image num is smaller and large epoch tends to overfit.

- Others
  - add perturb is helpful to the result but not too large.
  - add noise_std seems not help in psnr metric.
  - NeRF training is quite sensitive to the initialization, sometimes you need to restart the training if psnr is small in beginning.
  - Select the batch size and lrstep proportional to the num of images/rays of the training set.

- When implement nerf, I meet a very hard to find bug:
  - I first use `weights_coarse = weights[:, :self.get_ray_cfgs('n_sample')-2]` in fine resampling,  which use the end-pts weights
  as mid_pts weights, but this cause the PSNR drop by 4~5. 
  - Only `weights_coarse = weights[:, 1:self.get_ray_cfgs('n_sample')-1]` works, which take the front-pts weights.
  - Possibly this affects the correct sampling pts near surface.
