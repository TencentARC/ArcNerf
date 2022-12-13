// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// sampler related func


#include <torch/torch.h>

#include "utils.h"


// -------------------------------------------------- ------------------------------------ //


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
    torch::Tensor ray_numstep_counter);

void rays_sampler(
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

    // checking
    CHECK_INPUT(rays_o)
    CHECK_IS_FLOATING(rays_o)
    CHECK_INPUT(rays_d)
    CHECK_IS_FLOATING(rays_d)
    CHECK_INPUT(density_grid_bitfield)
    CHECK_INPUT(pts)
    CHECK_IS_FLOATING(pts)
    CHECK_INPUT(dirs)
    CHECK_IS_FLOATING(dirs)
    CHECK_INPUT(dt)
    CHECK_IS_FLOATING(dt)
    CHECK_INPUT(rays_numsteps)
    CHECK_IS_INT(rays_numsteps)
    CHECK_INPUT(ray_numstep_counter)
    CHECK_IS_INT(ray_numstep_counter)

    return rays_sampler_cuda(
        rays_o, rays_d, density_grid_bitfield, near_distance,
        n_sample, min_step, max_step, cone_angle, aabb0, aabb1, n_grid, n_cascades,
        pts, dirs, dt, rays_numsteps, ray_numstep_counter);
}


// -------------------------------------------------- ------------------------------------ //


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rays_sampler", &rays_sampler, "rays_sampler (CUDA)");
}
