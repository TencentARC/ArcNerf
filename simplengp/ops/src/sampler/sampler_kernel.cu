// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// sampler related func

#include <torch/extension.h>

#include "common.h"
extern pcg32 rng;


// -------------------------------------------------- ------------------------------------ //

__global__ void rays_sampler_cuda_kernel(
    const uint32_t n_rays,
    const Vector2f aabb_range,
    const float near_distance,
    const uint32_t n_sample,
    const float min_step,
    const float max_step,
    const float cone_angle,
    const uint32_t n_grid,
    const uint32_t n_cascades,
    const Vector3f *__restrict__ rays_o,
    const Vector3f *__restrict__ rays_d,
    const uint8_t *__restrict__ density_grid_bitfield,
    uint32_t *__restrict__ numsteps_counter,
    uint32_t *__restrict__ numsteps_out,
    Vector3f *__restrict__ pts,
    Vector3f *__restrict__ dirs,
    float *__restrict__ dt_out,
    default_rng_t rng)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n_rays)
        return;

    rng.advance(i * 8);

    Vector3f ray_o = rays_o[i];
    Vector3f ray_d = rays_d[i];

    Vector2f tminmax = aabb_ray_intersect(aabb_range, ray_o, ray_d);

    // The near distance prevents learning of camera-specific fudge right in front of the camera
    tminmax.x() = fmaxf(tminmax.x(), near_distance);
    float startt = tminmax.x();
    // add random step
    startt += calc_dt(startt, cone_angle, min_step, max_step) * random_val(rng);

    Vector3f idir = ray_d.cwiseInverse();

    // first pass to compute an accurate number of steps
    uint32_t j = 0;
    float t = startt;
    Vector3f pos;
    while (bbox_contains(aabb_range, pos = ray_o + t * ray_d) && j < n_sample)
    {
        float dt = calc_dt(t, cone_angle, min_step, max_step);
        uint32_t mip = mip_from_dt(dt, pos, n_grid, n_cascades);
        if (density_grid_occupied_at(pos, density_grid_bitfield, mip, n_grid))
        {
            ++j;
            t += dt;
        }
        else
        {
            uint32_t res = (n_grid >> mip);
            t = advance_to_next_voxel(t, cone_angle, min_step, max_step, pos, ray_d, idir, res);
        }
    }

    uint32_t numsteps = j;
    uint32_t base = atomicAdd(numsteps_counter, numsteps); // first entry in the array is a counter

    // move to base
    pts += base;
    dirs += base;
    dt_out += base;

    numsteps_out[2 * i + 0] = numsteps;
    numsteps_out[2 * i + 1] = base;
    if (j == 0) { return; }

    t = startt;
    j = 0;
    while (bbox_contains(aabb_range, pos = ray_o + t * ray_d) && j < numsteps)
    {
        float dt = calc_dt(t, cone_angle, min_step, max_step);
        uint32_t mip = mip_from_dt(dt, pos, n_grid, n_cascades);
        if (density_grid_occupied_at(pos, density_grid_bitfield, mip, n_grid))
        {
            // write the unwarp ones here
            pts[j] = pos;
            dirs[j] = ray_d;
            dt_out[j] = dt;
            ++j;
            t += dt;
        }
        else
        {
            uint32_t res = (n_grid >> mip);
            t = advance_to_next_voxel(t, cone_angle, min_step, max_step, pos, ray_d, idir, res);
        }
    }
}


void rays_sampler_cuda(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor density_grid_bitfield,
    const float near_distance,
    const int n_sample,
    const float min_step,
    const float max_step,
    const float cone_angle,
    const float aabb0,
    const float aabb1,
    const int n_grid,
    const int n_cascades,
    torch::Tensor pts,
    torch::Tensor dirs,
    torch::Tensor dt,
    torch::Tensor rays_numsteps,
    torch::Tensor ray_numstep_counter) {

    cudaStream_t stream=0;

    // input
    Vector2f aabb_range(aabb0, aabb1);
    Vector3f* rays_o_p = (Vector3f*)rays_o.data_ptr();
    Vector3f* rays_d_p = (Vector3f*)rays_d.data_ptr();
    uint8_t* density_grid_bitfield_p = (uint8_t*)density_grid_bitfield.data_ptr();

    // output
    Vector3f* pts_p = (Vector3f*)pts.data_ptr();
    Vector3f* dirs_p = (Vector3f*)dirs.data_ptr();
    float* dt_p = (float*)dt.data_ptr();
    uint32_t* rays_numsteps_p = (uint32_t*)rays_numsteps.data_ptr();
    uint32_t* ray_numstep_counter_p = (uint32_t*)ray_numstep_counter.data_ptr();

    const uint32_t n_rays = rays_o.sizes()[0];

    linear_kernel(rays_sampler_cuda_kernel, 0, stream, n_rays,
        aabb_range, near_distance, (uint32_t)n_sample, min_step, max_step, cone_angle,
        (uint32_t)n_grid, (uint32_t)n_cascades, rays_o_p, rays_d_p, density_grid_bitfield_p,
        ray_numstep_counter_p, rays_numsteps_p, pts_p, dirs_p, dt_p,
        rng);

    rng.advance();
    cudaDeviceSynchronize();
}
