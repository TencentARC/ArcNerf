# Camera
A Perspective Camera with `intrinsic(3x3)` and `c2w(4x4)` with get/set function.
- rescale: rescale based on image resize, only affect intrinsic.
- rescale_pose: rescale the c2w translation, only affect c2w translate.
- get_rays: get rays from cam, support index and random sample.
- proj_world_to_pixel: project points in world space to pixel.
- proj_world_to_cam: project points in world space to cam space.

# ray_helper
Ray function in rendering, or helper function for sampling. Other geometric function are in `geometry.ray`.
- get_rays: core get_ray function, lift pixel to world space and get rays_d(norm).
- get_near_far_from_rays: get near/far from rays/near/far/bounding_radius, used in models.
- get_zvals_from_near_far: get the zvals from near/far by sampling.
- sample_pdf: resample pts from pdf, will call `sample_cdf` after get cdf from weights.
- sample_cdf: resample pts from cdf.
- ray_marching: ray marching function and get color, will output intermediate result for debug.
- sample_ray_marching_output_by_index: sample ray marching intermediate result for 2d visual