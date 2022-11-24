// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// multivol bitfield related func in cuda. Some are actually same as bitfield func.


#include <torch/torch.h>

#include "utils.h"


// define the real cuda function to be called by c++ wrapper.
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
    const bool inclusive,
    torch::Tensor zvals,
    torch::Tensor mask);


/* c++ wrapper of sparse_volume_sampling forward func
   @param: rays_o, ray origin, (N_rays, 3)
   @param: rays_d, ray direction, assume normalized, (N_rays, 3)
   @param: near, near intersection zvals. (N_rays, 1)
   @param: far, far intersection zvals. (N_rays, 1)
   @param: N_pts, max num of sampling pts on each ray.
   @param: cone_angle: for mip stepping sampling. 0 means const dt
   @param: min_step: min stepping distance
   @param: max_step: max stepping distance
   @param: min_aabb_range, bbox range of inner volume, (2, 3) of xyz_min/max of inner volume
   @param: aabb_range, bbox range of volume, (2, 3) of xyz_min/max of whole volume
   @param: n_grid, num of grid
   @param: n_cascade, cascade level
   @param: bitfield, (n_grid**3 / 8) uint8 bit
   @param: near_distance, near distance for sampling. By default 0.0.
   @param: inclusive, whether to include in the inner volume. By default False
   @return: zvals, (N_rays, N_pts), sampled points zvals on each rays.
   @return: mask, (N_rays, N_pts), show whether each ray has intersection with the volume, BoolTensor
*/
void sparse_sampling_in_multivol_bitfield(
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
    const bool inclusive,
    torch::Tensor zvals,
    torch::Tensor mask){
    // checking
    CHECK_INPUT(rays_o)
    CHECK_IS_FLOATING(rays_o)
    CHECK_INPUT(rays_d)
    CHECK_IS_FLOATING(rays_d)
    CHECK_INPUT(near)
    CHECK_IS_FLOATING(near)
    CHECK_INPUT(far)
    CHECK_IS_FLOATING(far)
    CHECK_INPUT(min_aabb_range)
    CHECK_IS_FLOATING(min_aabb_range)
    CHECK_INPUT(aabb_range)
    CHECK_IS_FLOATING(aabb_range)
    CHECK_INPUT(bitfield)
    CHECK_IS_BYTE(bitfield)
    CHECK_INPUT(zvals)
    CHECK_IS_FLOATING(zvals)
    CHECK_INPUT(mask)
    CHECK_IS_BOOL(mask)

    if (rays_o.size(1) != 3 || rays_d.size(1) != 3) {
        throw std::runtime_error{"Input rays tensor must be (B, 3)."};
    }

    if (near.size(1) != 1 || far.size(1) != 1) {
        throw std::runtime_error{"Input near/far tensor must be (B, 1)."};
    }

    if (min_aabb_range.size(0) != 2 || min_aabb_range.size(1) != 3) {
        throw std::runtime_error{"min xyz range should be in (2, 3)."};
    }

    if (aabb_range.size(0) != 2 || aabb_range.size(1) != 3) {
        throw std::runtime_error{"xyz range should be in (2, 3)."};
    }

    int n_level = inclusive ? n_cascade : n_cascade-1;
    if (bitfield.size(0) != (n_grid * n_grid * n_grid / 8) * n_level) {
        throw std::runtime_error{"bitfield should be in (n_grid**3/8*n_level,)."};
    }


    if (zvals.size(0) != rays_o.size(0) || zvals.size(1) != n_pts) {
        throw std::runtime_error{"zval should be in (n_rays, n_pts)."};
    }

    if (mask.size(0) != rays_o.size(0) || mask.size(1) != n_pts) {
        throw std::runtime_error{"mask should be in (n_rays, n_pts)."};
    }

    // call actual cuda function
    return sparse_sampling_in_multivol_bitfield_cuda(
        rays_o, rays_d, near, far, n_pts, cone_angle, min_step, max_step,
        min_aabb_range, aabb_range, n_grid, n_cascade, bitfield, near_distance, inclusive,
        zvals, mask
    );
}


// -------------------------------------------------- ------------------------------------ //


void generate_grid_samples_multivol_cuda(
    const torch::Tensor density_grid,
    const int density_grid_ema_step,
    const int n_elements,
    const torch::Tensor aabb_range,
    const int n_cascade,
    const int n_grid,
    const float thresh,
    const bool inclusive,
    torch::Tensor density_grid_positions_uniform,
    torch::Tensor density_grid_indices_uniform);

void generate_grid_samples_multivol(
    const torch::Tensor density_grid,
    const int density_grid_ema_step,
    const int n_elements,
    const torch::Tensor aabb_range,
    const int n_cascade,
    const int n_grid,
    const float thresh,
    const bool inclusive,
    torch::Tensor density_grid_positions_uniform,
    torch::Tensor density_grid_indices_uniform) {

    // checking
    CHECK_INPUT(density_grid)
    CHECK_IS_FLOATING(density_grid)
    CHECK_INPUT(aabb_range)
    CHECK_IS_FLOATING(aabb_range)
    CHECK_INPUT(density_grid_positions_uniform)
    CHECK_IS_FLOATING(density_grid_positions_uniform)
    CHECK_INPUT(density_grid_indices_uniform)
    CHECK_IS_INT(density_grid_indices_uniform)

    return generate_grid_samples_multivol_cuda(
        density_grid, density_grid_ema_step, n_elements, aabb_range, n_cascade, n_grid, thresh, inclusive,
        density_grid_positions_uniform, density_grid_indices_uniform);
}


// -------------------------------------------------- ------------------------------------ //


void update_bitfield_multivol_cuda(
    const torch::Tensor density_grid,
    const float density_grid_mean,
    torch::Tensor density_grid_bitfield,
    const float thres,
    const int n_grid,
    const int n_cascade,
    const bool inclusive);

void update_bitfield_multivol(
    const torch::Tensor density_grid,
    const float density_grid_mean,
    torch::Tensor density_grid_bitfield,
    const float thres,
    const int n_grid,
    const int n_cascade,
    const bool inclusive) {

    // checking
    CHECK_INPUT(density_grid)
    CHECK_IS_FLOATING(density_grid)
    CHECK_INPUT(density_grid_bitfield)

    return update_bitfield_multivol_cuda(
        density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid, n_cascade, inclusive);
}



// -------------------------------------------------- ------------------------------------ //



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_grid_samples_multivol", &generate_grid_samples_multivol, "generate_grid_samples_multivol (CUDA)");
    m.def("update_bitfield_multivol", &update_bitfield_multivol, "update_bitfield_multivol (CUDA)");
    m.def("sparse_sampling_in_multivol_bitfield", &sparse_sampling_in_multivol_bitfield, "sparse_sampling_in_multivol_bitfield (CUDA)");
}
