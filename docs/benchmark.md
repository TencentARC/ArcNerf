# Benchmark on some common dataset

The benchmark is only for view synthesis, which evals the synthesized image psnr.

-----------------------------------------------------------------------
Expname are in the format of `{dataset}_{scene}_{model}_{other_settings}`.


## NeRF and related
### Lego  (800x800, 25 eval images)

| Method |        cfg         | PSNR |    Official repo   |    Official PSNR     | paper PSNR  | Others |
|:------:|:------------------:|:----:|:------------------:|:--------------------:|:-----------:|:-------|
|  NeRF  |configs/expr/NeRF/lego/nerf_lego_nerf.yaml|32.86|https://github.com/yenchenlin/nerf-pytorch|32.3|32.54|  |
|  NeuS  |configs/expr/NeRF/lego/nerf_lego_neus.yaml|30.81|https://github.com/Totoro97/NeuS| 31.12 |  NA |embed_pts=10 following official repo|
| VolSDF |configs/expr/NeRF/lego/nerf_lego_volsdf.yaml|28.25| https://github.com/lioryariv/volsdf | 20.77 |NA| Official repo not converge well on lego scene  |
|MipNeRF |configs/expr/NeRF/lego/nerf_lego_mipnerf.yaml|34.19| https://github.com/google/mipnerf | NA |35.74| TODO: Not fully match up yet|

* NeRF: We have another repo contains only the function for vanilla nerf. You can visit [simplenerf](https://github.com/TencentARC/simplenerf) for more detail.

-----------------------------------------------------------------------

#### Instant-NGP on lego
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

Our result:
- On white background. Use customize volume in torch(Not many cuda code).
- Original repo converge much faster. Since we have lots of over-head here for pipeline modularity.

| Num steps | time | PSNR | comment   |
|:---------:|:----:|:----:|:---------:|
| 100 | ~4s  | 14.17 | Crop stage, not converge well|
| 500 | ~18s | 17.07 | Crop stage, not converge  well|
| 2k  | ~40s | 26.42 |  |
| 1w  | ~3min| 31.99 |  |
| 5w  | ~17min  | 34.65 | |

* Many factor that could affect the result(Like using `black background` improve PSNR to `~35.0`.)
* We implement most of the operation in torch rather than Highly optimized CUDA kernels. It is more flexible for experiment but slower in speed.
* We have another repo contains only the function for instant-ngp, which runs faster(`~15 min`) and better than this repo. It contains functions for ngp only.
It uses more CUDA implementation from original repo. You can visit [simplengp](https://github.com/TencentARC/simplengp) for more detail and expr log.
* In our framework, we can easily plugin the `HashEncoder` or `SparseVolumeSample` for other models(eg. `NeuS`).

-----------------------------------------------------------------------
### Full Benchmark on NeRF synthetic dataset

All image with white-bkg, same as the eval in vanilla NeRF.

|          |   chair    |   drums    |   ficus    |   hotdog   |   lego     | materials  |    mic     |   ship     |     |   avg  |
|:--------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:---:|:------:|
|nerf     |   33.30    |   25.11    |   30.47    |   36.73    |   32.86    |   29.87    |   **29.75**    |   28.70    |  | | |
|neus     |   31.36    |   24.26    |   25.86    |   36.66    |   30.72    |   29.09    |   30.50    |   26.41    | | 29.358 |
|mipnerf  |   34.41    |   25.45    |   32.95    |   37.45    |   35.36    |   30.70    |   34.84    |   29.83    | | 32.624 |
|nerf_ngp |   34.14    |   25.33    |   30.20    |   36.11    |   34.24    |   28.30    |   34.90    |   28.19    | | 31.426 |
|neus_ngp |   31.64    |   22.81    |   26.05    |   32.09    |   30.51    |   25.29    |   27.54    |   24.19    | | 27.515 |

* The volume in nerf/neus_ngp is simply volume with `side=2.0`. More accurate 3d volume bbox generally leads to better performance.

-----------------------------------------------------------------------

## LLFF
### Fern  (378*504, 3 eval images)
We only use non-ndc version. For ndc space, you need to refer to our [simplenerf](https://github.com/TencentARC/simplenerf) project.

| Method |        cfg         | PSNR |    Official repo   |    Official PSNR     | paper PSNR  |
|:------:|:------------------:|:----:|:------------------:|:--------------------:|:-----------:|
|NeRF(non-ndc)|configs/expr/LLFF/fern/llff_fern_nerf.yaml|26.17|https://github.com/yenchenlin/nerf-pytorch|26.29(non-ndc)|NA|


## Benchmark on LLFF Forward-face dataset
All run for 20w iter and in non ndc-space.

|          |    fern    |   flower   |  fortress  |   horns    |   leaves   |  orchids   |    room    |   trex     |     |  avg  |
|:--------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:---:|:------:|
|nerf      |   26.17    |   27.19    |   27.03    |   27.20    |   19.78    |   20.34    |   31.82    |   26.38    |     | 25.822 |
|mipnerf   |   25.41    |   27.41    |   29.12    |   28.01    |   20.17    |   20.17    |   30.63    |   26.40    |     | 25.915 |


-----------------------------------------------------------------------

## DTU
DTU has 15 indoor scene with limit background. Since mask is provided, we change the background to white like `NeRF` dataset,
and eval on the pure object image with white bkg. Testhold out is 8 (`1/8` images are for test and not in train).

|         |   24  |   37  |   40  |   55  |   63  |   65  |   69  |   83  |   97  |  105  |  106  |  110  |  114  |  118  |  122  |     |  avg |
|:-------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---:|:----:|
|nerf     | 27.26 | 26.05 | 27.68 | **24.21** | 27.14 | 25.82 | 22.16 | 28.13 | 25.30 | 28.28 | 23.12 | 26.79 | 27.73 | 26.88 | 28.52 |     |      |
|neus     | 27.12 | 26.23 | 27.98 | 27.52 | 29.85 | 26.03 | 23.75 | 26.78 | 25.79 | 28.48 | 23.38 | 26.48 | 27.70 | 24.74 | 30.72 |     |      |
|mipnerf  | 27.08 | **24.73** | **25.78** | 28.18 | 28.78 | **20.81** | 23.11 | 28.30 | 26.17 | 28.93 | **22.99** | 27.23 | 27.25 | 27.29 |**00.00**|
|nerf_ngp | 27.16 | 26.48 | 28.96 | 28.93 | 30.78 | 27.10 | 24.01 | 30.41 | 26.21 | 30.15 | 24.55 | 28.32 | 28.69 | 27.77 | 32.54 |     |
|neus_ngp | 26.61 | 24.91 | 27.31 | 28.43 | 28.89 | 25.57 | 23.04 | 26.51 | 24.84 | 27.92 | 22.48 | 25.20 | 27.56 | 24.28 | 30.56 |     |

* mipnerf sometimes converge fast as 10w iter in DTU. Maybe the image num is smaller.
* The volume in neus/nerf_ngp are set to `side=1.5` without tuning for each case specifically, only `scan=24` we use `side=2.0`.
The result could be better for more accurate volume selection.

-----------------------------------------------------------------------

## TanksAndTemplates


-----------------------------------------------------------------------

## Capture
### qqtiger
It is a more daily scene captured by us. It reflects the algorithm performance on common daily scenes.
It contains a foreground object and background.
