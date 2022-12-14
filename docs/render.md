# Camera
A Perspective Camera with `intrinsic(3x3)` and `c2w(4x4)` with get/set function.
- rescale: rescale based on image resize, only affect intrinsic.
- rescale_pose: rescale the c2w translation, only affect c2w translate.
- get_rays: get rays from cam, support index and random sample.
- proj_world_to_pixel: project points in world space to pixel.
- proj_world_to_cam: project points in world space to cam space.
- Allow you to set/get the `intrinsic/c2w/w2c` matrix easily.

------------------------------------------------------------------------
# ray_helper
Ray function in rendering, or helper function for sampling. Other geometric function are in `geometry.ray`.
- get_rays: core get_ray function, lift pixel to world space and get rays_d(norm). TAKE CARE ABOUT THE FLATTEN ORDER!
  - NDC space is allowed to get rays for unbounded cases.
  - Following MipNeRF, it returns a radius of rays on the pixel.
  - center_pixel: If use center-pixel(like `instant-ngp`/`mip-nerf`, it will use center-pixel from (0.5, 1.5...) to generate 3d points).
- get_near_far_from_rays: get near/far from rays/near/far/bounding_radius, used in models.
- get_zvals_from_near_far: get the zvals from near/far by sampling.
- get_zvals_from_near_far_fix_step: get the zvals using a fix step from near to far. A `mask_pts` is given for invalid pts.
- get_zvals_outside_sphere/get_zvals_from_sphere_radius: Get zvals from sphere radius(use in background like `mpi`).
- sample_pdf: resample pts from pdf, will call `sample_cdf` after get cdf from weights.
- sample_cdf: resample pts from cdf.
- ray_marching: ray marching function and get color, will output intermediate result for debug.
- sample_ray_marching_output_by_index: sample ray marching intermediate result for 2d visual
- make_sample_rays: fake ray-points generator.


![march](../assets/ray_helper/march.png)
![sample](../assets/ray_helper/sample.png)

------------------------------------------------------------------------
# render
Several renderers have been provided for rendered mesh/color mesh.
## open3d
Only use for geometry rendering, no backward is allowed. Can not be used for training.
## pytorch3d
Batch process, allow mesh/color mesh rendering, can be used for training.
## nvdiffrast
https://github.com/NVlabs/nvdiffrast

To be implemented.
