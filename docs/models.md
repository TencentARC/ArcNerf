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
## implicit network
implicit network for mapping xyz -> sigma/sdf/rgb, etc.
### GeoNet
Multiple DenseLayer/SirenLayer. For details, ref to the implementation.
### RadianceNet
Multiple DenseLayer/SirenLayer. For details, ref to the implementation.

# chunk_size
The model is hard to process `batch_size * n_rays_per_sample * n_pts_per_ray` in a single
forward, set it to be `n_rays` to be process in a single forward.
By default use `1024*32 = 32768` rays together.

For input, dataset generate `(B, N_rays, ...)`, where `B` is the num of image/sample in a batch, `N_rays` is num of
rays from each sample. We flat samples in batch and get `(B*N_rays, ...)` together to send to the network. Those
rays are processed in chunk and output return in `(B*N_rays, ...)` as well. Finally, we reshape them and
get `(B, N_rays, ...)` as final results.

The main `forward` function in `Base3dModel` is a wrapper for `(B, N_rays, ...)` input, call `_forward` by `chunk_rays`
size,  and the core `_forward` in child class (like `NeRF`) process `(N_rays_per_chunk, ...)` and get result for rays.

In the core `_forward`, the model may forward the pts sampled on rays. We set `chunk_pts` for a single forward size for
pts. By default we use `4096*192=786432`, it works good for 32GB memory GPU.

## Rays
The dataset only provides `rays_o` and `rays_d`, but the actual sampling procedure is in model. Dataset may provide
`bounds` for sampling guidance, which is generally coming from point_cloud in cam space.
- near: Hard reset the near zvals for all rays
- far: Hard reset the far zvals for all rays
- bounding_radius: If not None, will use to calculate near/far in the sphere.
But it could be overwrite by hardcode near/far.
- bounds: If bounds is provided in dataset, use it instead of bounding radus.
But it could be overwrite by hardcode near/far.
- n_sample: Init sample point.
- n_importance: Point sampled from hierarchical sampling
- add_inf_z: When True, will use inf_z at last for raymarching, needed for rays including background range.
    - If you add a separate background model, you should not use it, so that ray inside the sphere focus on the object.
- noise_std: if >0.0, add to sigma when ray marching. good for training.
- perturb: perturb zvals during training.
- inverse_linear: If True, more points are sampled closer to near zvals.

## background
There are three ways to handle background
- (1) Set a far zvals in rays for sampling. It will combine obj+background together for rendering.
`ImgLoss` can be applied on the whole image for optimization.
- (2) Constrain the far zvals by bounds or bounding_radius. Use `white_bkg`, then all the background will be in white.
`MaskImgLoss` needs to be applied to get obj area optimized. `MaskLoss` can also be applied for geometry.
- (3) Use a separate background model(nerf++), restrict the inner rays in sphere. Combine the inner and background model
For color.
`ImgLoss` and be applied on the combined image. `MaskLoss` and `MaskImageLoss` can be applied on obj image.

## Nerf
Nerf model with single forward(NeRF), and hierarchical sampling(NeRFFull).
It is combination of GeoNet and RadianceNet, with ray sampling, resample pdf, ray marching, etc.
