# Benchmark on some common dataset

The benchmark is only for view synthesis, which evals the synthesized image psnr.

-----------------------------------------------------------------------
Expname are in the format of `{dataset}_{scene}_{model}_{other_settings}`.


## NeRF
### Lego  (800x800, 25 eval images)

| Method |        cfg         | PSNR |    Official repo   |    Official PSNR     | paper PSNR  | Others |
|:------:|:------------------:|:----:|:------------------:|:--------------------:|:-----------:|:-------|
|  NeRF  |configs/expr/NeRF/lego/nerf_lego_nerf.yaml|32.86|https://github.com/yenchenlin/nerf-pytorch|32.3|32.54|  |
|  NeuS  |configs/expr/NeRF/lego/nerf_lego_neus.yaml|30.78|https://github.com/Totoro97/NeuS| 31.12 |  NA |-30.84 in smaller sphere /-embed_pts=10 following official repo|
| VolSDF |configs/expr/NeRF/lego/nerf_lego_volsdf.yaml|28.14| https://github.com/lioryariv/volsdf | 20.77 |NA| Official repo not converge well on lego scene  |
|MipNeRF |configs/expr/NeRF/lego/nerf_lego_mipnerf.yaml|32.93| https://github.com/google/mipnerf | TODO |35.74||

#### Instant-NGP
The highly optimized [instant-ngp](https://github.com/NVlabs/instant-ngp) model, official performance:
- max_samples: 1024, color space: sRGB, max_res: 524288(per_level_scale: 2.0)

| Num steps | time | PSNR in linear space | PSNR in sRGB space|
|:---------:|:----:|:--------------------:|:-----------------:|
| 100 | ~1s | 21.46 | 21.79 |
| 500 | ~5s | 29.62 | 29.83 |
| 2k  | ~15s| 33.42 | 33.67 |
| 1w  |~1min| 35.67 | 35.11 |
| 5w  |~5min| 36.36 | 35.78 |


## LLFF
### Fern  (378*504, 3 eval images)
We only use non-ndc version.

| Method |        cfg         | PSNR |    Official repo   |    Official PSNR     | paper PSNR  |
|:------:|:------------------:|:----:|:------------------:|:--------------------:|:-----------:|
|NeRF(non-ndc)|configs/expr/LLFF/fern/llff_fern_nerf.yaml|26.17|https://github.com/yenchenlin/nerf-pytorch|26.29(non-ndc)|NA|


-----------------------------------------------------------------------

# Time and speed
