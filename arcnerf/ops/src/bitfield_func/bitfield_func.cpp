// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// bitfield related func in cuda


#include <torch/torch.h>

#include "utils.h"


// define the real cuda function to be called by c++ wrapper.
void sparse_volume_sampling_bit_cuda(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor near,
    const torch::Tensor far,
    const int n_pts,
    const float dt,
    const torch::Tensor aabb_range,
    const int n_grid,
    const torch::Tensor bitfield,
    const float near_distance,
    torch::Tensor zvals,
    torch::Tensor mask);


/* c++ wrapper of sparse_volume_sampling forward func
   @param: rays_o, ray origin, (N_rays, 3)
   @param: rays_d, ray direction, assume normalized, (N_rays, 3)
   @param: near, near intersection zvals. (N_rays, 1)
   @param: far, far intersection zvals. (N_rays, 1)
   @param: N_pts, max num of sampling pts on each ray.
   @param: dt, fix step length
   @param: aabb_range, bbox range of volume, (2, 3) of xyz_min/max of whole volume
   @param: n_grid, num of grid
   @param: bitfield, (n_grid**3 / 8) uint8 bit
   @param: near_distance, near distance for sampling. By default 0.0.
   @return: zvals, (N_rays, N_pts), sampled points zvals on each rays.
   @return: mask, (N_rays, N_pts), show whether each ray has intersection with the volume, BoolTensor
*/
void sparse_volume_sampling_bit(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor near,
    const torch::Tensor far,
    const int n_pts,
    const float dt,
    const torch::Tensor aabb_range,
    const int n_grid,
    const torch::Tensor bitfield,
    const float near_distance,
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

    if (aabb_range.size(0) != 2 || aabb_range.size(1) != 3) {
        throw std::runtime_error{"xyz range should be in (2, 3)."};
    }

    if (bitfield.size(0) != (n_grid * n_grid * n_grid / 8)) {
        throw std::runtime_error{"bitfield should be in (n_grid**3/8,)."};
    }

    if (zvals.size(0) != rays_o.size(0) || zvals.size(1) != n_pts) {
        throw std::runtime_error{"zval should be in (n_rays, n_pts)."};
    }

    if (mask.size(0) != rays_o.size(0) || mask.size(1) != n_pts) {
        throw std::runtime_error{"mask should be in (n_rays, n_pts)."};
    }

    // call actual cuda function
    return sparse_volume_sampling_bit_cuda(
        rays_o, rays_d, near, far, n_pts, dt, aabb_range, n_grid, bitfield, near_distance,
        zvals, mask
    );
}


// -------------------------------------------------- ------------------------------------ //


void generate_grid_samples_cuda(
    const torch::Tensor density_grid,
    const int density_grid_ema_step,
    const int n_elements,
    const int n_grid,
    const float thresh,
    torch::Tensor density_grid_positions_uniform,
    torch::Tensor density_grid_indices_uniform);

void generate_grid_samples(
    const torch::Tensor density_grid,
    const int density_grid_ema_step,
    const int n_elements,
    const int n_grid,
    const float thresh,
    torch::Tensor density_grid_positions_uniform,
    torch::Tensor density_grid_indices_uniform) {

    // checking
    CHECK_INPUT(density_grid)
    CHECK_IS_FLOATING(density_grid)
    CHECK_INPUT(density_grid_positions_uniform)
    CHECK_IS_FLOATING(density_grid_positions_uniform)
    CHECK_INPUT(density_grid_indices_uniform)
    CHECK_IS_INT(density_grid_indices_uniform)

    return generate_grid_samples_cuda(
        density_grid, density_grid_ema_step, n_elements, n_grid, thresh,
        density_grid_positions_uniform, density_grid_indices_uniform);
}


// -------------------------------------------------- ------------------------------------ //


void splat_grid_samples_cuda(
    const torch::Tensor density,
    const torch::Tensor density_grid_indices,
    const int n_density_grid_samples,
    torch::Tensor density_grid_tmp);

void splat_grid_samples(
    const torch::Tensor density,
    const torch::Tensor density_grid_indices,
    const int n_density_grid_samples,
    torch::Tensor density_grid_tmp) {

    // checking
    CHECK_INPUT(density)
    CHECK_IS_FLOATING(density)
    CHECK_INPUT(density_grid_indices)
    CHECK_IS_INT(density_grid_indices)
    CHECK_INPUT(density_grid_tmp)
    CHECK_IS_FLOATING(density_grid_tmp)

    return splat_grid_samples_cuda(
        density, density_grid_indices, n_density_grid_samples, density_grid_tmp);

}


// -------------------------------------------------- ------------------------------------ //


void ema_grid_samples_nerf_cuda(
    const torch::Tensor density_grid_tmp,
    int n_elements,
    float decay,
    torch::Tensor density_grid);

void ema_grid_samples_nerf(
    const torch::Tensor density_grid_tmp,
    int n_elements,
    float decay,
    torch::Tensor density_grid) {

    // checking
    CHECK_INPUT(density_grid_tmp)
    CHECK_IS_FLOATING(density_grid_tmp)
    CHECK_INPUT(density_grid)
    CHECK_IS_FLOATING(density_grid)

    return ema_grid_samples_nerf_cuda(density_grid_tmp, n_elements, decay, density_grid);
}


// -------------------------------------------------- ------------------------------------ //


void update_bitfield_cuda(
    const torch::Tensor density_grid,
    const float density_grid_mean,
    torch::Tensor density_grid_bitfield,
    const float thres,
    const int n_grid);

void update_bitfield(
    const torch::Tensor density_grid,
    const float density_grid_mean,
    torch::Tensor density_grid_bitfield,
    const float thres,
    const int n_grid) {

    // checking
    CHECK_INPUT(density_grid)
    CHECK_IS_FLOATING(density_grid)
    CHECK_INPUT(density_grid_bitfield)

    return update_bitfield_cuda(density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid);
}

// -------------------------------------------------- ------------------------------------ //


void count_bitfield_cuda(const torch::Tensor density_grid_bitfield, torch::Tensor counter, const int n_grid);

void count_bitfield(const torch::Tensor density_grid_bitfield, torch::Tensor counter, const int n_grid) {
    return count_bitfield_cuda(density_grid_bitfield, counter, n_grid);
}



// -------------------------------------------------- ------------------------------------ //


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_grid_samples", &generate_grid_samples, "generate_grid_samples (CUDA)");
    m.def("sparse_volume_sampling_bit", &sparse_volume_sampling_bit, "sparse volume sampling in bitfield (CUDA)");
    m.def("splat_grid_samples", &splat_grid_samples, "splat_grid_samples (CUDA)");
    m.def("ema_grid_samples_nerf", &ema_grid_samples_nerf, "ema_grid_samples_nerf (CUDA)");
    m.def("update_bitfield", &update_bitfield, "update_bitfield (CUDA)");
    m.def("count_bitfield", &count_bitfield, "count_bitfield (CUDA)");
}
