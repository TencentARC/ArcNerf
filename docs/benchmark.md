# Benchmark on some common dataset

The benchmark is only for view synthesis, which evals the synthesized image psnr.

-----------------------------------------------------------------------
Expname are in the format of `{dataset}_{scene}_{model}_{other_settings}`.


## NeRF
### Lego  (800x800, 25 eval images)

| Method |        cfg         | PSNR |    Official repo   |    Official PSNR     | paper PSNR  | Others |
|:------:|:------------------:|:----:|:------------------:|:--------------------:|:-----------:|:-------|
|  NeRF  |configs/expr/NeRF/lego/nerf_lego_nerf.yaml|32.86|https://github.com/yenchenlin/nerf-pytorch|32.3|32.54|  |
|  NeuS  |configs/expr/NeRF/lego/nerf_lego_neus.yaml|30.81|https://github.com/Totoro97/NeuS| 31.12 |  NA |embed_pts=10 following official repo|
| VolSDF |configs/expr/NeRF/lego/nerf_lego_volsdf.yaml|28.25| https://github.com/lioryariv/volsdf | 20.77 |NA| Official repo not converge well on lego scene  |
|MipNeRF |configs/expr/NeRF/lego/nerf_lego_mipnerf.yaml|32.95| https://github.com/google/mipnerf | TODO |35.74| TODO: Not match up yet|

#### Instant-NGP
The highly optimized [instant-ngp](https://github.com/NVlabs/instant-ngp) model, official performance:
- max_samples: 1024, color space: sRGB, max_res: 2048, lr=1e-2
- The original eval uses black background, but original `NeRF` uses white bkg

| Num steps | time | PSNR in sRGB space | PSNR in linear space| PSNR in white bkg, sRGB | PSNR in white bkg, linear |
|:---------:|:----:|:--------------------:|:-----------------:|:-----------------:|:-----------------:|
| 100 | ~1s | 21.46 | 21.79 | 21.49 | 21.59 |
| 500 | ~5s | 29.62 | 29.83 | 29.34 | 29.02 |
| 2k  | ~15s| 33.42 | 33.67 | 33.06 | 32.73 |
| 1w  |~1min| 35.67 | 35.11 | 35.22 | 34.61 |
| 5w  |~5min| 36.36 | 35.78 | 36.02 | 35.12 |


## LLFF
### Fern  (378*504, 3 eval images)
We only use non-ndc version. For ndc space, you need to refer to our [simplenerf](https://github.com/TencentARC/simplenerf) project.

| Method |        cfg         | PSNR |    Official repo   |    Official PSNR     | paper PSNR  |
|:------:|:------------------:|:----:|:------------------:|:--------------------:|:-----------:|
|NeRF(non-ndc)|configs/expr/LLFF/fern/llff_fern_nerf.yaml|26.17|https://github.com/yenchenlin/nerf-pytorch|26.29(non-ndc)|NA|


-----------------------------------------------------------------------

# Time and speed
