# Trails on Lego Scene
Here are the trails and experience on Lego scene.


# NeRF (density model)

For some basic setting like `lr`/`precrop`/`maskloss`, you can visit our project [simplenerf](http://github.com/TencentARC/simplenerf) for more experiment log.
But what worth notice is:
- precrop is important for such white_bkg case since nerf model is sensitive to initialization.

Thanks to our pipeline, we can easily change any part in the pipeline and see each what `pruning`/`embedding` contribute to the result:

Basic NeRF PSNR on Lego is `32.86`. [conf](../configs/expr/NeRF/lego/nerf_lego_nerf.yaml)

## In data preperation
- directly add `center_pixel` for ray generation improves NeRF(`freq embed`) performance(`~1`).  [conf](../configs/expr/NeRF/lego/trails/nerf_lego_nerf_centerpixel.yaml)

## embedding level
We can change `freqEncode` into `tcnn.HashGridEncoding/SHEncoding`.

- Using `ngp` like encodings, you must use `center_pixel`. Otherwise not converge. (This has also been proven in `ngp` model.)
Without using a shallow network, we can get PSNR `27.82`. [conf](../configs/expr/NeRF/lego/trails/nerf_lego_nerf_ngpembed_centerpixel_trunc.yaml).
It happens that geometry on object is good but noisy in empty space is heavy.

![nerf_ngpembed](../assets/expr/nerf_ngpembed.png)

- Further, decrease the mlp size using a shallow MLP make the speed faster (`0.3 -> 0.05s/iter`). The object is converage well but empty space is even noisier.
[conf](../configs/expr/NeRF/lego/trails/nerf_lego_nerf_ngpembed_centerpixel_shallow_trunc.yaml).

![nerf_ngpembed_shallow](../assets/expr/nerf_ngpembed_shallow.png)

## Volume Pruning
Based on original NeRF implementation, we can keep `freq embed`, can add volume structure with pruning to improve modelling and performance.
With such object structure, we can set the `n_sample` large(1024), and it will remain samples based on volume voxels.

- With directly sample 1024 pts and volume structure, speed and results both increase (`0.04s` + `PSNR 33.33`). [conf](../configs/expr/NeRF/lego/trails/nerf_lego_nerf_volumeprune_moresample_noimportance.yaml).

- If we use original 64 + 128 sampling method in NeRF, the result is lower to `27.57`, meaning that the coarse sampling is not accurate to produce `resample` points. [conf](../configs/expr/NeRF/lego/trails/nerf_lego_nerf_volumeprune.yaml).

## ngp
`instant-ngp` combines volume pruning with hash/sh encoding, for much faster converge.
You can visit our project [simplengp](http://github.com/TencentARC/simplengp) for more experiment log.

- Without the `fusemlp` but used original mlp with same size, the PSNR drop by `~0.4`. Not sure whether `bias` or other factors affect the result.
- ngp model can extract correct geometry, but color is wrong compared to NeRF model(we use the face normal as directly). If you need to ngp model for
geometry extraction, you need to optimize the texture map in other way.
- resample more pts near surface can further improve the PSNR(`~0.3`), but time for each step will increase by `x2`. [conf](../configs/expr/NeRF/lego/trails/nerf_lego_nerf_ngp_resample.yaml)
But this still can not make the extraction color correct.


------------------------------------------------------------------------------------------------------------



# MipNerf (for novel view rendering)
For mip-Nerf, it is hard to used for object extraction, but provide better rendering result.
Here are the trails and experience on Lego scene.

- `center_pixel` is necessary in `mipnerf` which is the same as official repo.


------------------------------------------------------------------------------------------------------------



# NeuS (sdf model)

Basic NeRF PSNR on Lego is `32.78`. [conf](../configs/expr/NeRF/lego/nerf_lego_nerf.yaml)

------------------------------------------------------------------------------------------------------------


# volsdf (sdf model)

- Running


------------------------------------------------------------------------------------------------------------
Summary:

|          model            | PSNR  |  iter/s | Num iter | eval s/img | conf file|
|:-------------------------:|:-----:|:-------:|:--------:|:----------:|:--------:|
| NeRF                      | 32.86 | 0.25s   |  30w     | 10s        | [conf](../configs/expr/NeRF/lego/nerf_lego_nerf.yaml) |
|+center_pixel              | 33.80 | 0.25s   |  30w     | 10s        | [conf](../configs/expr/NeRF/lego/trails/nerf_lego_nerf_centerpixel.yaml)
|  +hash/sh encoder           | 27.82 | 0.32s   |  30w     | 10s        | [conf](../configs/expr/NeRF/lego/trails/nerf_lego_nerf_ngpembed_centerpixel_trunc.yaml)
|  +hash/sh encoder + shallow | 9.27 | 0.05s   |  30w     | 1.3s       | [conf](../configs/expr/NeRF/lego/trails/nerf_lego_nerf_ngpembed_centerpixel_shallow_trunc.yaml)
|+volume pruning(1024 pts)  | 33.33 | 0.04s   |  30w     | 0.84s      | [conf](../configs/expr/NeRF/lego/trails/nerf_lego_nerf_volumeprune_moresample_noimportance.yaml)
|+volume pruning(64 + 128 pts)  | 27.57 | 0.2s   |  30w     | 4.13s      | [conf](../configs/expr/NeRF/lego/trails/nerf_lego_nerf_volumeprune.yaml)
| ngp                       | 34.31 | 0.018s   |  5w      | 0.24s     | [conf](../configs/expr/NeRF/lego/nerf_lego_nerf_ngp.yaml)
|    + new volume           | 34.65 | 0.017s   |  5w      | 0.24s     | [conf](../configs/expr/NeRF/lego/trails/nerf_lego_nerf_ngp_newvolume.yaml)



------------------------------------------------------------------------------------------------------------

# Inference on result

## Extraction
We use a pre-defined volume, and use each voxel's center point to extract density/sdf, then apply marching cube
to get the mesh.

For the color on mesh, we use the triangle centroid as pts, with -face_norm as directly to get the color for face.

- Advance algorithm for texture map/lighting are not supported. You many possibly need to do diffRendering to optimize
the texture and other asset used for modern graphical engine. We will try to do it in the future.

- Although we support customized volume for extraction(the volume has same side, or different lenght on xyz dimension).
But we found that only xyz with same length could generate color correctly.

TODO: Add nerf/neus geo/color mesh output images!!!
