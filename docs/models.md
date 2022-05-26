# Base_modules
Modules like embedder, implicit function, radiance field, etc.
## embedder
Embed inputs like xyz/view_dir into higher dimension using periodic funcs(sin, cos).
- output_dim = (inputs_dim * len(periodic_fns) * N_freq + include_input * inputs_dim)
- log_sampling: if True, use log factor sin(2**N * x). Else use scale factor sin(N * x).
## activation
- Sine: sin() for activation.
- get_activation: get the activation by cfg
## linear
- DenseLayer: Linear with custom activation function.
- SirenLayer: Linear with sin() activation and respective initialization.

## BaseGeoNet/BaseRadianceNetwork
This models serve as the basic block for modeling obj geometry and radiance. It can be pure linear networks, pure volume,
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

### Dense Volume Model
Explicit volume for mapping xyz -> sigma/sdf/rgb, etc, using voxel interpolation. Pure volume or mix of volume/network.
Specify the type as `VolGeoNet/VolRadianceNet`. We support dense volume with pruning in torch implementation.
#### VolGeoNet
Volume with value/feature on grid, network is optional to used. For details, ref to the implementation.
- geometric_init: If True, init the geonet such that represent a sdf of sphere with radius_init(inner sdf < 0).
- W_feat_vol: If this > 0, the model interpolate feature from grid_pts. With a dense implementation, it could be slow.
Try to not use W_feat_vol(use network with xyz input, or directly use pure volume).
#### RadianceNet
Volume with value/feature on grid, network is optional to used. For details, ref to the implementation.
## use_nn
Set this to make linear layers after volume grid output. But it could increase the time used.

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

# Rays
The dataset only provides `rays_o` and `rays_d`, but the actual sampling procedure is in model. Dataset may provide
`bounds` for sampling guidance, which is generally coming from point_cloud in cam space.
- near: Hard reset the near zvals for all rays
- far: Hard reset the far zvals for all rays
- bounding_radius: If not None, will use to calculate near/far in the sphere.
But it could be overwrite by hardcode near/far.
- bounds: If bounds is provided in dataset, use it instead of bounding radius.
- But it could be overwrite by hardcode near/far.

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

# background
There are four ways to handle background
- (1) Set a far zvals in rays for sampling. It will combine obj+background together for rendering.
`ImgLoss` can be applied on the whole image for optimization.
  - background will be noisy and hard to model.
- (2) Constrain the far zvals by bounds or bounding_radius. `MaskImgLoss` needs to be applied to get obj area optimized.
`MaskLoss` should also be applied for geometry.
- (3) If the obj does not have mask, but it is with white background(like nerf `lego`  dataset`).
Set `white_bkg` in the rays, and sample in the ball. Directly compare the image and output rgb.
- (4) Use a separate background model(nerf++), restrict the inner rays in sphere. Combine the inner and background model
For color. `ImgLoss` and be applied on the combined image. `MaskLoss` and `MaskImageLoss` can be applied on obj image.

The first three methods do not require a separate model, but the final one does.

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

## Base_3d_Model
Base_3d_Model is the class for all fg_model and bkg_model. bkg_model is able to used as fg_model
if you actually need it. It generally contains geo_net and radiance_net for geometry/rgb reconstrunction.

## fg_model
Here are the class of fg_model, which concentrates on building the foreground object part.
### Nerf
Nerf model with single forward(NeRF), and hierarchical sampling(NeRFFull).

It is combination of GeoNet and RadianceNet, with ray sampling, resample pdf, ray marching, etc.
### Neus
Neus model sdf as geo value and up-sample pts.

It is combination of GeoNet and RadianceNet, with up sample, resample pdf, ray marching, etc.

Since it gets sdf value instead of sigma, we do not support sigma mode for blending.
(Actuall we can do, but `rgb blending` has better result)

- init_var: use to init the inv_s param. By default `inv_s = -np.log(init_var)/speed_factor`, init as `0.05`
- speed_factor: use to init the inv_s, and get scale by `exp(inv_s * speed_factor``. By default `10`.
- anneal_end: num of epoch for slope blending. In infer model, the factor is `1`, else `min(epoch/anneal_end, 1)`


## bkg_model
Here are the class of bkg_model, which concentrates on building the background.
The model is also able to model foreground together if you set the parameters well.
### NeRFPP(nerf++)
The nerf++ model use same structure of one stage NeRF model to model the background in Multi-Sphere Image(MSI),
and change input to `(x/r, y/r, z/r, 1/r)` for different radius.
