# Benchmark on some common dataset

The benchmark is only for view synthesis, which evals the synthesized image psnr.

Based on the type of data, we could benchmark on :
    - object level for `NeRF`/`DTU`/`NSVF`/`RTMV`
    - scene level for `LLFF`/`MipNeRF360`/`TanksAndTemplate`/`Capture`/`BlendedMVS`
    - We have run on some dataset for benchmark, but not all. We will try to run on all datasets when time and resource allowed.

-----------------------------------------------------------------------
Expname are in the format of `{dataset}_{scene}_{model}_{other_settings}`.


## NeRF and related
We follow the exact same setting on dataset split as original NeRF implementation at https://github.com/yenchenlin/nerf-pytorch.
### Lego  (800x800, 25 eval images)

| Method |        cfg         | PSNR |    Official repo   |    Official PSNR     | paper PSNR  | Others |
|:------:|:------------------:|:----:|:------------------:|:--------------------:|:-----------:|:-------|
|  NeRF  |configs/expr/NeRF/lego/nerf_lego_nerf.yaml|32.86|https://github.com/yenchenlin/nerf-pytorch|32.3|32.54|  |
|  NeuS  |configs/expr/NeRF/lego/nerf_lego_neus.yaml|30.81|https://github.com/Totoro97/NeuS| 31.12 |  NA |embed_pts=10 following official repo|
| VolSDF |configs/expr/NeRF/lego/nerf_lego_volsdf.yaml|28.25| https://github.com/lioryariv/volsdf | 20.77 |NA| Official repo not converge well on lego scene  |
|MipNeRF |configs/expr/NeRF/lego/nerf_lego_mipnerf.yaml|35.36| https://github.com/google/mipnerf | NA |35.74| TODO: Not fully match up yet|

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
| 100 | ~4s  | 16.11 | Crop stage, not converge well|
| 500 | ~18s | 17.95 | Crop stage, not converge  well|
| 2k  | ~40s | 30.01 |  |
| 1w  | ~3min| 33.14 |  |
| 5w  | ~17min  | 35.38 | |

* We found that large lr and loss weight leads to sharp sigma distribution and final PSNR, so we use `lr=1e-1 & loss_weight=3000`.
* Many factor that could affect the result(Like using `black background` improve PSNR to `~35.78`.)
* We implement most of the operation in torch rather than Highly optimized CUDA kernels. It is more flexible for experiment but slower in speed.
* We have another repo contains only the function for instant-ngp. It contains functions for ngp only and gets better result.
It uses more CUDA implementation from original repo. You can visit [simplengp](https://github.com/TencentARC/simplengp) for more detail and expr log.
* In our framework, we can easily plugin the `HashEncoder` or `SparseVolumeSample` for other models(eg. `NeuS`).

-----------------------------------------------------------------------
### Full Benchmark on NeRF synthetic dataset

All image with white-bkg, same as the eval in vanilla NeRF.

|          |   chair    |   drums    |   ficus    |   hotdog   |   lego     | materials  |    mic     |   ship     |     |   avg  |
|:--------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:---:|:------:|
|nerf     |   33.30    |   25.11    |   30.47    |   36.73    |   32.86    |   29.87    |   33.24    |   28.70    | | 31.285 |
|neus     |   31.36    |   24.26    |   25.86    |   36.66    |   30.72    |   29.09    |   30.50    |   26.41    | | 29.358 |
|mipnerf  |   34.41    |   25.45    |   32.95    |   37.45    |   35.36    |   30.70    |   34.84    |   29.83    | | 32.624 |
|nerf_ngp |   34.88    |   25.50    |   30.55    |   36.92    |   35.38    |   29.12    |   34.80    |   28.39    | | 31.942 |
|neus_ngp |   31.64    |   22.81    |   26.05    |   32.09    |   30.51    |   25.29    |   27.54    |   24.19    | | 27.515 |

* The volume in nerf/neus_ngp is simply volume with `side=2.0`. More accurate 3d volume bbox generally leads to better performance.

-----------------------------------------------------------------------

## LLFF
We follow the exact same setting on dataset split as original NeRF implementation at https://github.com/yenchenlin/nerf-pytorch.
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
and eval on the pure object image with white bkg.

For the dataset split, We do not follow the exact setting as NeuS but keep `testhold` out is `8` (`1/8` images are for test and not in train).

|         |   24  |   37  |   40  |   55  |   63  |   65  |   69  |   83  |   97  |  105  |  106  |  110  |  114  |  118  |  122  |     |  avg |
|:-------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---:|:----:|
|nerf     | 27.26 | 26.05 | 27.68 | 24.21 | 27.14 | 25.82 | 22.16 | 28.13 | 25.30 | 28.28 | 23.12 | 26.79 | 27.73 | 26.88 | 28.52 |     | 26.338 |
|neus     | 27.12 | 26.23 | 27.98 | 27.52 | 29.85 | 26.03 | 23.75 | 26.78 | 25.79 | 28.48 | 23.38 | 26.48 | 27.70 | 24.74 | 30.72 |     | 26.837 |
|mipnerf  | 27.08 | 26.23 | 25.78 | 28.18 | 28.78 | 26.09 | 23.11 | 28.30 | 26.17 | 28.93 | 23.70 | 27.23 | 27.25 | 27.29 | 31.49 |     | 27.041 |
|nerf_ngp | 27.16 | 26.48 | 28.96 | 28.93 | 30.78 | 27.10 | 24.01 | 30.41 | 26.21 | 30.15 | 24.55 | 28.32 | 28.69 | 27.77 | 32.54 |     | 28.137 |
|neus_ngp | 26.61 | 24.91 | 27.31 | 28.43 | 28.89 | 25.57 | 23.04 | 26.51 | 24.84 | 27.92 | 22.48 | 25.20 | 27.56 | 24.28 | 30.56 |     | 26.274 |

* mipnerf sometimes converge fast as 10w iter in DTU. Maybe the image num is smaller.
* The volume in neus/nerf_ngp are set to `side=1.5` without tuning for each case specifically, only `scan=24` we use `side=2.0`.
The result could be better for more accurate volume selection.


-----------------------------------------------------------------------

## MipNeRF360

MipNeRF360 has 9 outdoor 360 scene with background. We run on the public available 7 scenes.

For the dataset split, we do not follow the exact setting as MipNeRF360 but keep `testhold` out is `8` (`1/8` images are for test and not in train, roughly 25+).

Images are resize by `1/4` on each dimension for training and testing.

### garden
We benchmark on the garden split for now. We can model the fg/bkg separately, or model the whole scene

|     Method      | iter | PSNR |           cfg       | Modeling | details|
|:---------------:|:----:|:----:|:-------------------:|:--------:|:------:|
|       NeRF      | 30w  |      |configs/expr/MipNeRF360/garden/mipnerf360_garden_nerf.yaml| fg only | nerf with 64+128 sampling|
|      NeRF++     | 30w  |      |configs/expr/MipNeRF360/garden/mipnerf360_garden_nerf_nerfpp.yaml| fg(nerf) + bkg(nerf++/msi) | fg/bkg each 32+64 sample |
|     MipNeRF     | 50w  |      |configs/expr/MipNeRF360/garden/mipnerf360_garden_nerf.yaml| fg_only | mipnerf with 128+128 sample |
|    Neus+NeRF++  | 30w  |      |configs/expr/MipNeRF360/garden/mipnerf360_garden_nerf.yaml| fg(neus) + bkg(nerf++/msi) | fg 64+64, bkg 32+64 sample |
|     Multivol    | 5w   |      |configs/expr/MipNeRF360/garden/mipnerf360_garden_multivol.yaml| fg_only | multivol sampling with inner volume, 1024 sampling, that is the `instant-ngp` method |
|  nerfngp+multivol| 5w  |      |configs/expr/MipNeRF360/garden/mipnerf360_garden_nerfngp_multivol.yaml| fg(nerf_ngp) + bkg(multivol) | fg/bkg each 1024 sample with different hashEnc+mlp |
|nerfngp+multivol+sigma_blending|  5w |      |configs/expr/MipNeRF360/garden/mipnerf360_garden_nerfngp_multivol_sigma.yaml| fg(nerf_ngp) + bkg(multivol) | same as last but get blending results by merging sigmas and render |
| neusngp+multivol |  5w |      |configs/expr/MipNeRF360/garden/mipnerf360_garden_neusngp_multivol.yaml| fg(neus_ngp) + bkg(multivol)  | fg/bkg each 1024 sample with different hashEnc+mlp, fg is a neus |

- For the methods modeling fg + bkg separately, the samples are in two splits. Each split gets a rgb value, and fg model gets transmittance as well. `full_color = fg_color + T * bkg_color`. It could be seen that
the boundary area between fg/bkg are not that clear compared to directly modeling fg+bkg together. But separate modeling generally leads to better foreground result.
- The multivol/ngp model sometimes get `inf` grad on `hashenc`, use grad_clipping to forbid.
- For the scenes with sky, you should set `white_bkg: True  # sky` for the bkg_model to avoid empty black color.
- The initialization is not that stable, sometimes you need to run several times to get optimized solution.

-----------------------------------------------------------------------

## TanksAndTemplates

TanksAndTemplates has 8 outdoor 360 scene with background. But since the official [link](https://www.tanksandtemples.org/) do not contain intrinsic parameters,
we use the processed on at [nerf++](https://github.com/Kai-46/nerfplusplus), which has 4 scenes instead.

For the dataset split, we follow the same split in `nerf++`, where the train/test ratio is roughly 8. Image size are the same.



-----------------------------------------------------------------------

## Capture
We only provide a real captured data call `qqtiger` at [here](../data/qqtiger.MOV). It tells you how to use the real captured data.
For data preprocessing of it, please vis [doc](./datasets.md) or run
`python tools/extract_video.py --configs configs/datasets/Capture/qqtiger.yaml`
`python tools/run_poses.py --configs configs/datasets/Capture/qqtiger.yaml`.
Remember to set your only path.

We provide the precessed pose file at [data](../data/poses_bounds.npy.zip). You can put them together in to
`your_data_dir/Capture/qqtiger/` and run.
### qqtiger
It is a more daily scene captured by us. It reflects the algorithm performance on common daily scenes.
It contains a foreground object and background.
