// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// multivol bitfield related func in cuda


#include <torch/extension.h>

#include "common.h"
#include "volume_func.h"
#include "pcg32.h"


// ------------------------------------------------------------------------------------------------ //


// The real cuda kernel
__global__ void sparse_sampling_in_multivol_bitfield_cuda_kernel(
    const uint32_t n_rays,
    const Vector3f *__restrict__ rays_o,  //(N_rays, 3)
    const Vector3f *__restrict__ rays_d,  //(N_rays, 3)
    const float *__restrict__ near,  //(N_rays, 1)
    const float *__restrict__ far,  //(N_rays, 1)
    const Vector3f *__restrict__ min_aabb_range,  // (2, 3)
    const Vector3f *__restrict__ aabb_range,  // (2, 3)
    const uint8_t *__restrict__ bitfield,  //(n_grid**3)
    const uint32_t n_grid,
    const uint32_t n_cascade,
    const uint32_t n_pts,
    const float cone_angle,
    const float min_step,
    const float max_step,
    const float near_distance,
    default_rng_t rng,
    float *__restrict__ zvals,  //(N_rays, N_pts)
    bool *__restrict__ mask) {  //(N_rays, N_pts)

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;  // ray id
    if (i >= n_rays) return;

    rng.advance(i * 8);

    Vector3f _rays_o = rays_o[i];
    Vector3f _rays_d = rays_d[i];

    float startt = fmaxf(near[i], near_distance);
    float far_end = far[i];
    // perturb init zvals
    startt += calc_dt(startt, cone_angle, min_step, max_step) * random_val(rng);

    const Vector3f xyz_min = aabb_range[0];
    const Vector3f xyz_max = aabb_range[1];
    const Vector3f minvol_xyz_min = min_aabb_range[0];
    const Vector3f minvol_xyz_max = min_aabb_range[1];

    uint32_t j = 0;
    float t = startt;
    Vector3f pos;

    // move zvals and mask
    zvals += i * n_pts;
    mask += i * n_pts;

    while (t <= far_end && j < n_pts && check_pts_in_aabb(pos = _rays_o + _rays_d * t, xyz_min, xyz_max))
    {
        float dt = calc_dt(t, cone_angle, min_step, max_step);
        uint32_t mip = mip_from_pos(pos, minvol_xyz_min, minvol_xyz_max, n_cascade);

        if (mip == 0) {  // since it hits the inner volume, revert back
            j = 0;  // The sampling should not cover bkg -> fg -> bkg sequence
            t = advance_to_next_voxel(t, dt, pos, _rays_d, minvol_xyz_min, minvol_xyz_max, n_grid);
        } else {
            if (density_grid_occupied_at_multivol(pos, bitfield, mip, minvol_xyz_min, minvol_xyz_max, n_grid)) {
                // update the pts and mask
                zvals[j] = t;
                mask[j] = true;
                ++j;
                t += dt;
            } else {  // advance by min vol size, pos is normed now
                t = advance_to_next_voxel_multivol(t, cone_angle, min_step, max_step, pos, _rays_d, minvol_xyz_min, minvol_xyz_max, n_grid);
            }

        }
    }

    // make the remaining zvals the same as last
    float last_zval;
    if (j > 0 && j < n_pts) {
        last_zval = zvals[j-1];
    }
    while (j > 0 && j < n_pts) {
        zvals[j] = last_zval;
        ++j;
    }

    return;
}


/* CUDA instantiate of sparse_volume_sampling func
   @param: rays_o, ray origin, (N_rays, 3)
   @param: rays_d, ray direction, assume normalized, (N_rays, 3)
   @param: near, near intersection zvals. (N_rays, 1)
   @param: far, far intersection zvals. (N_rays, 1)
   @param: n_pts, max num of sampling pts on each ray.
   @param: cone_angle: for mip stepping sampling. 0 means const dt
   @param: min_step: min stepping distance
   @param: max_step: max stepping distance
   @param: min_aabb_range, bbox range of inner volume, (2, 3) of xyz_min/max of inner volume
   @param: aabb_range, bbox range of volume, (2, 3) of xyz_min/max of each volume
   @param: n_grid, num of grid
   @param: n_cascade, cascade level
   @param: bitfield, (n_grid**3 / 8) uint8 bit
   @param: near_distance, near distance for sampling. By default 0.0.
   @return: zvals, (N_rays, N_pts), sampled points zvals on each rays.
   @return: mask, (N_rays, N_pts), show whether each ray has intersection with the volume, BoolTensor
*/
void sparse_sampling_in_multivol_bitfield_cuda(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor near,
    const torch::Tensor far,
    const int n_pts,
    const float cone_angle,
    const float min_step,
    const float max_step,
    const torch::Tensor min_aabb_range,
    const torch::Tensor aabb_range,
    const int n_grid,
    const int n_cascade,
    const torch::Tensor bitfield,
    const float near_distance,
    torch::Tensor zvals,
    torch::Tensor mask) {

    cudaStream_t stream=0;

    const uint32_t n_rays = rays_o.size(0);  // N_rays

    // inputs
    Vector3f* rays_o_p = (Vector3f*)rays_o.data_ptr();
    Vector3f* rays_d_p = (Vector3f*)rays_d.data_ptr();
    Vector3f* min_aabb_range_p = (Vector3f*)min_aabb_range.data_ptr();
    Vector3f* aabb_range_p = (Vector3f*)aabb_range.data_ptr();
    float* near_p = (float*)near.data_ptr();
    float* far_p = (float*)far.data_ptr();
    uint8_t* bitfield_p = (uint8_t*)bitfield.data_ptr();

    // outputs
    float* zvals_p = (float*)zvals.data_ptr();
    bool* mask_p = (bool*)mask.data_ptr();

    linear_kernel(sparse_sampling_in_multivol_bitfield_cuda_kernel, 0, stream, n_rays,
        rays_o_p, rays_d_p, near_p, far_p, min_aabb_range_p, aabb_range_p, bitfield_p,
        (uint32_t)n_grid, (uint32_t)n_cascade, (uint32_t)n_pts, cone_angle, min_step, max_step, near_distance,
        rng, zvals_p, mask_p);

    rng.advance();
    cudaDeviceSynchronize();

}


// -------------------------------------------------- ------------------------------------ //
