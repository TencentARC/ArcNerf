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


## LLFF
### Fern  (378*504, 3 eval images)
We only use non-ndc version.

| Method |        cfg         | PSNR |    Official repo   |    Official PSNR     | paper PSNR  |
|:------:|:------------------:|:----:|:------------------:|:--------------------:|:-----------:|
|NeRF(non-ndc)|configs/expr/LLFF/fern/llff_fern_nerf.yaml|26.17|https://github.com/yenchenlin/nerf-pytorch|26.29(non-ndc)|NA|


-----------------------------------------------------------------------

# Time and speed
