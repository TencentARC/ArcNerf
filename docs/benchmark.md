# Benchmark on some common dataset

-----------------------------------------------------------------------
Expname are in the format of `{dataset}_{scene}_{model}_{other_settings}`.


## NeRF
### Lego  (800x800, 25 eval images)

| Method |        cfg         | PSNR |    Official repo   |    Official PSNR     | paper PSNR  |
|:------:|:------------------:|:----:|:------------------:|:--------------------:|:-----------:|
|  NeRF  |configs/expr/NeRF/lego/nerf_lego_nerf.yaml|31.28(100K)|https://github.com/yenchenlin/nerf-pytorch|32.3|32.54|
|  NeuS  |configs/expr/NeRF/lego/nerf_lego_neus.yaml|    | https://github.com/Totoro97/NeuS | 31.12  |  NA |
| VolSDF |    |    |   |     |


## LLFF
### Fern  (378*504, 3 eval images)
We only use non-ndc version.

| Method |        cfg         | PSNR |    Official repo   |    Official PSNR     | paper PSNR  |
|:------:|:------------------:|:----:|:------------------:|:--------------------:|:-----------:|
|NeRF(non-ndc)|configs/expr/LLFF/fern/llff_fern_nerf.yaml|25.47(100k)|https://github.com/yenchenlin/nerf-pytorch|26.29|NA|



# Time and speed
