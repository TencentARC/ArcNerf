# Base_modules
Modules like embedder, implicit function, radiance field, etc.

## activation
- Sine: sin() for activation.
- get_activation: get the activation by cfg
## linear
- DenseLayer: Linear with custom activation function.
- SirenLayer: Linear with sin() activation and respective initialization.

------------------------------------------------------------------------
## encoder
We provide several encoder to transfer the input xyz/dir into higher freq embeddings.

Some customized cuda implementations needed to be installed by `sh scripts/install_ops.sh`
### FreqEmbedder
Positional encoding introduced in [NeRF](https://arxiv.org/abs/2003.08934).

Embed inputs like xyz/view_dir into higher dimension using periodic funcs(sin, cos).
### GaussianEmbedder
Integrated Positional encoding introduced in [MipNeRF](https://arxiv.org/abs/2103.13415).

You need to first get the Gaussian representation of the interval(mean/cov), then embed it into higher freq.
### HashGridEmbedder
The multi-res hashing embedding introduced in [Instant-ngp](https://arxiv.org/abs/2201.05989).

It only embeds xyz positions rather than direction. It uses multi-res volume grid index, hash them and get the
embedding of grid_pts from hashmap, and get the embedding by interpolation.

You can select the backend by setting `backend`=`torch`/`cuda`/`tcnn`.
### SHEmbedder
The spherical harmonic hashing embedding introduced in [Instant-ngp](https://arxiv.org/abs/2201.05989).

It only embeds xyz direction rather than positions.

You can select the backend by setting `backend`=`torch`/`cuda`/`tcnn`.
### DenseGridEmbedder
The dense grid embedder directly extracts density and feature from a dense volume. It only embeds xyz direction rather than positions.

## Tiny-cuda-nn
Some encoders are implemented in [tiny-cuda-nn]() by Nvidia. You should clone the repo `--recursive`
and install them by `sh scripts/install_tinycudann.sh`.

------------------------------------------------------------------------
## obj_bound
This class serves for the foreground modeling that it uses a geometric structure(`volume`,`sphere`) to restrict the sampling
on pts in a smaller region. In some case(like `volume`), the structure is allowed to be pruned based on volume density,
and a more accurate and sparse sampling can be performed.

This structure bounds the real object so that the fg_model can focus on modelling the object. Rays that does not hit
the structure can be skipped from computation.

You should set `model.obj_bound` to set the bounding structure.

### BasicBound
It does not contain any structure, and the sampling is performing based on ray configs. Points are either in a near/far
range, or bounding by a sphere with larger radius that can cover all cameras.

### VolumeBound
It contains a dense volume with bitfield to record the density in every voxel. Optimization can be performed if you set.
The points then will be sampled in remaining voxel. And rays computation can be largely reduced in coarse structure.
- origin/n_grid/xyz_len/side: The information of the volume.

### SphereBound
It contains a smaller sphere bounding the real object. In sdf modeling like `Neus`, it samples inside such a sphere.
- origin/radius: The information of the sphere.

------------------------------------------------------------------------
## BaseGeoNet/BaseRadianceNetwork
These models serve as the basic block for modeling obj geometry and radiance. It can be pure linear networks, pure volume,
mix of volume and network, sparse octree, multi-res volume with hashing, tensorRF etc. All the blocks can be select from
`geometry/radiance` part in `cfgs.model`.

### Linear Network Model
implicit network for mapping xyz -> sigma/sdf/rgb, etc. Pure linear network
Specify the type as `GeoNet/RadianceNet`(or leave it blank, which will be default)
#### GeoNet
Multiple DenseLayer/SirenLayer. For details, ref to the implementation.
- geometric_init: If True, init the geonet such that represent a sdf of sphere with radius_init(inner sdf < 0).
siren layer will use pretrain, dense layer will use weight/bias init(But like an oval than sphere).
#### RadianceNet
Multiple DenseLayer/SirenLayer. For details, ref to the implementation.

------------------------------------------------------------------------
# chunk_size
## chunk_rays
The model is hard to process `batch_size * n_rays_per_sample * n_pts_per_ray` in a single
forward, set it to be `n_rays` to be process in a single forward.
By default use `1024*32 = 32768` rays together.

For input, dataset generate `(B, N_rays, ...)`, where `B` is the num of image/sample in a batch, `N_rays` is num of
rays from each sample. We flat samples in batch and get `(B*N_rays, ...)` together to send to the network. Those
rays are processed in chunk and output return in `(B*N_rays, ...)` as well. Finally, we reshape them and
get `(B, N_rays, ...)` as final results.

The main `forward` function in `Base3dModel` is a wrapper for `(B, N_rays, ...)` input, call `_forward` by `chunk_rays`
size,  and the core `_forward` in child class (like `NeRF`) process `(N_rays_per_chunk, ...)` and get result for rays.

## chunk_pts
In the core `_forward`, the model may forward the pts sampled on rays. We set `chunk_pts` for a single forward size for
pts. By default we use `4096*192=786432`, it works good for 32GB memory GPU.

## gpu_online
The chunk process function supports bringing tensor online for each chunk and bring back after processing.
This helps to save GPU memory in case large batch size is used and brought to GPU for calculation and concat.

You just need to keep the tensors in cpu and set `gpu_on_func` and it will bring every tensor to GPU online.
(But large concatenation still takes time.)

Generally we don't use this mode but put everything on GPU together. You need to carefully specify the chunk size
when input size is huge(like 512^3).

------------------------------------------------------------------------
# Model Configs
To see the model configs, you can go to `configs/models`. They are also used in unittest to check the correctness and give
information.

## Obj_bound
Ref to the `obj_bound` section above.

## Rays
The dataset only provides `rays_o` and `rays_d`, but the actual sampling procedure is in model. Dataset may provide
`bounds` for sampling guidance, which is generally coming from point_cloud in cam space.
- near: Hard reset the near zvals for all rays
- far: Hard reset the far zvals for all rays
- bounding_radius: If not None, will use to calculate near/far in the sphere.
But it could be overwritten by hardcode near/far.
- bounds: If bounds is provided in dataset, use it instead of bounding radius.
- But it could be overwritten by hardcode near/far.

For point sampling:
- n_sample: Init sample point.
- n_importance: Point sampled from hierarchical sampling
- perturb: perturb zvals during training.
- inverse_linear: If True, more points are sampled closer to near zvals.

For ray marching(color blending):
- add_inf_z: When True, will use inf_z at last for raymarching, needed for rays including background range.
    - If you add a separate background model, you should not use it, so that ray inside the sphere focus on the object.
- noise_std: if >0.0, add to sigma when ray marching. good for training.
- white_bkg: If True, will make the rays with mask = 0 as rgb = 1.0

------------------------------------------------------------------------
# background
There are four ways to handle background
- (1) Set a far zvals in rays for sampling. It will combine obj+background together for rendering.
`ImgLoss` can be applied on the whole image for optimization.
  - background will be noisy and hard to model.
- (2) Constrain the far zvals by bounds or bounding_radius. `MaskImgLoss` needs to be applied to get obj area optimized.
`MaskLoss` should also be applied for geometry.
- (3) If the obj does not have mask, but it is with white background(like nerf `lego` dataset).
Set `white_bkg` in the rays, and sample in the ball. Directly compare the image and output rgb.
- (4) Use a separate background model(nerf++), restrict the inner rays in sphere. Combine the inner and background model
For color. `ImgLoss` and be applied on the combined image. `MaskLoss` and `MaskImageLoss` can be applied on obj image.

The first three methods do not require a separate model, but the final one does.

------------------------------------------------------------------------
# Models
Below are the models we support.
- inference_only: Use in eval/infer mode. If True, only keep 'rgb/depth/mask',
            will remove progress item like 'sigma/radiance' to save memory.
- get_progress: Use in train/val mode. If True, keep 'sigma/radiance' for visual purpose.
- cur_epoch/total_epoch: to adjust strategies during training

## FullModel
Full Model is the class for all models. It contains fg_model and optional bkg_model.
It combines values from both model and get final results. If you don't use bkg model, you need
to reconstruct the whole scene by fg_model.
- The input size(like rays_o/rays_d) are `(B, N_rays, ...)` in `FullModel.forward()`.
But they are flattened into `(B*N_rays, ...)` and process by chunk in fg_model/bkg_model.
### bkg_blend
Method two merge bkg_model with fg_mdeol.
- If `rgb`, get fg and bkg rgb separately, then use fg_weight factor to mix bkg color.
  - In this mode, `add_inf_z` in `background.rays` should be True to get inf zvals.
  `add_inf_z` in `model.rays` must be False to avoid inf zvals in fg color computation.
- If `sigma`, merge all sigma from fg and bg together and do ray marching for val.
  - In this mode `add_inf_z` in `background.rays` must be False to get correct zvals to merge.
  `add_inf_z` in `model.rays` is suggested be True to avoid merge inf zvals for color computation.

------------------------------------------------------------------------
# Base_3d_Model
Base_3d_Model is the class for all fg_model and bkg_model. bkg_model is able to used as fg_model
if you actually need it. It generally contains geo_net and radiance_net for geometry/rgb reconstrunction.

------------------------------------------------------------------------
## fg_model
Here are the class of fg_model, which concentrates on building the foreground object part. You can choose to bound
the main object in a volume/sphere for accurate and concentrate sampling, or without the structure to sampling in large space.
- obj_bound: You can set a volume/sphere that bounding the obj. Look above section.
  - We may set up the automatic helper in the future. To find a close bounding structure.

- mask_rays:
  - Using bounding structure may reduce the calculation on some useless rays. You need to assign default values to this
  rays.

- mask_pts:
  - Use sparse structure like pruned volume make some ray sampling sparse. Some pts are not in the occupied voxels
  will be masked as useless. The fg model provides a wrapper `get_density_radiance_by_mask_pts` to do `_forward_pts_dir`
  with only masked points, it helps to save computation on useless pts.
  - For the mask, they are packed to `[T, T, T, F, F, F]` with `False` value only at the end of each ray. This will
  not harm the following raymarching/upsample funcitno.

- Optimization: The optimization is only for volume now. The params are under `cfgs.models.obj_bound`
  - epoch_optim: If not None, will set up the voxel occupancy and do pruning every this epoch.
  - epoch_optim_warmup: If not None, will do different sampling in volume.
  - ray_sampling_acc: If True, will do customized skip sampling in CUDA. Otherwise use simple uniform sampling in (near, far)
  - ray_sampling_fix_step: If True, use fix step to init all sample rather than uniformly sample in (near, far).
                           It reduces sample num with less computation.
  - ema_optim_decay: If None, directly write all `non-negative` opacity value by new one. Else, update old one by ema factor.
  - opa_thres: The minimum opacity for considering a voxel as occupied.

- default values
  - If you use a obj_bound structure to bound the object, many rays may not hit the structure so that they can
  be skipped for computation. You need to set up a default value for them.
  - bkg_color: color for invalid rays. float in `0~1`. By default `(1.0, 1.0, 1.0)`, white.
  - depth_far: depth value for invalid rays. By default `10.0`.
  - normal: normal for invalid rays. float in `0~1`. By default `(0.0, 1.0, 0.0)`, up direction.
  - progress: For the progress(pts in rays), it will all be set with zeros values.

Following are real modeling methods:

### NeRF
[NeRF](https://arxiv.org/abs/2003.08934) model with single forward or hierarchical sampling. You can control by `n_importance`.

It is combination of GeoNet and RadianceNet, with ray sampling, resample pdf, ray marching, etc.

### MipNeRF
[MipNerf](https://arxiv.org/abs/2103.13415) model do not use the isolated sampled points, but use a gaussian expectation for modeling the interval.
It is used for view synthesis rather than geometry extraction.

### sdf_model
SDF model is a class of fg_model that model the volumetric field as sdf. Compared to density field, it is better to
extract geometric information of the object.
- For sdf, we are easy to touch the surface. Instead of volume rendering, we provide a simple usage of surface rendering
by finding the surface xyz directly and use surface pts as the pixel rgb.
A simple tutorial is in `notebooks/surface_render.ipynb`.

#### Neus
[Neus](http://arxiv.org/abs/2106.10689) models sdf as geo value and up-sample pts.

It is combination of GeoNet and RadianceNet, with up sample, resample pdf, ray marching, etc.

Since it gets sdf value instead of sigma, we do not support sigma mode for blending.
(Actuall we can do, but `rgb blending` has better result)

- init_var: use to init the inv_s param. By default `inv_s = -np.log(init_var)/speed_factor`, init as `0.05`
- speed_factor: use to init the inv_s, and get scale by `exp(inv_s * speed_factor``. By default `10`.
- anneal_end: num of epoch for slope blending. In infer model, the factor is `1`, else `min(epoch/anneal_end, 1)`
- n_iter: num of iter to run upsample algo.
- radius_bound: This is the interest of radius that we bound.

### VolSDF
[VolSDF](https://arxiv.org/abs/2106.12052) models sdf as well but used different sdf_to_density function and sampling method compared with NeuS.
- init_beta: use to init the ln_beta param. By default `inv_s = np.log(init_beta)/speed_factor`, init as `0.1`
- speed_factor: use to init the inv_s, and get scale by `exp(ln_beta * speed_factor``. By default `10`.
- beta_min: min beta offset, by default 1e-4
- n_iter: num of iter to run upsample algo.
- n_eval: num of eval pts for upsampling, not used for backward.
- beta_iter: num of iter to update beta
- eps: small threshold
- radius_bound: This is the interest of radius that we bound.

The performance is worse than Neus as we test.

### Instant-ngp
[Instant-ngp](https://arxiv.org/abs/2201.05989) is not a model. It uses `HashGridEmbedder` and `SHEmbedder` to
accelerate the training progress. For volume-based acceleration, you should set `obj_bound` as a volume, use do
optim/ray_sample_acc for fast sampling.
- cascade: Since we focus on modeling the main object in the scene, we do not use multi-cascade for sampling
and density update. Only a single `N_grid**3` density bitfield is used for record and sampling.
- As in original repo, the data are processed such that all sampling ray points are in `[0, 1]` already. We
use the volume not normalized as a `[0, 1]` cuda but with customized position and length. The sampling is based
on the customized volume.

------------------------------------------------------------------------
## bkg_model
Here are the class of bkg_model, which concentrates on building the background.
The model is also able to model foreground together if you set the parameters well.

### NeRFPP(nerf++)
The [nerf++](http://arxiv.org/abs/2010.07492) model use same structure of one stage NeRF model to model the background in Multi-Sphere Image(MSI),
and change input to `(x/r, y/r, z/r, 1/r)` for different radius.
