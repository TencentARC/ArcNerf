# Trails on Lego Scene
Here are the trails and experience on Lego scene.


# NeRF (density model)

For some basic setting like `lr`/`precrop`/`maskloss`, you can visit our project [simplenerf](http://github.com/TencentARC/simplenerf) for more experiment log.
But what worth notice is:
- precrop is important for such white_bkg case since nerf model is sensitive to initialization.

Thanks to our pipeline, we can easily change any part in the pipeline and see each what `pruning`/`embedding` contribute to the result:

Basic NeRF PSNR on Lego is `32.78`. [conf](../configs/expr/NeRF/lego/nerf_lego_nerf.yaml)

## In data preperation
- directly add `center_pixel` for ray generation improves NeRF(`freq embed`) performance(`~1`).  [conf](../configs/expr/NeRF/lego/trails/nerf_lego_nerf_centerpixel.yaml)

## embedding level
We can change `freqEncode` into `tcnn.HashGridEncoding/SHEncoding`.

- Using `ngp` like encodings, you must use `center_pixel`. Otherwise not converge. (This has also been proven in `ngp` model.)
Without using a shallow network, we can get PSNR `xx`. [conf](../configs/expr/NeRF/lego/trails/nerf_lego_nerf_ngpembed_centerpixel.yaml)
- Further, decrease the mlp size using a shallow MLP


## Volume Pruning
Based on original NeRF implementation, we can keep `freq embed`, can add volume structure with pruning to improve modelling and performance.


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

|          model            | PSNR  |  iter/s | eval s/img |
|:-------------------------:|:-----:|:-------:|:----------:|
| NeRF                      | 32.78 |
|+center_pixel              | 33.79 |


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
