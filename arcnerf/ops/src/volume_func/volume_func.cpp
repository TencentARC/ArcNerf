// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// volume related func in cuda


#include <torch/torch.h>

#include "utils.h"


// define the real cuda function to be called by c++ wrapper.
torch::Tensor check_pts_in_occ_voxel_cuda(
    const torch::Tensor xyz,
    const torch::Tensor bitfield,
    const torch::Tensor range,
    const uint32_t n_grid);


/* c++ wrapper of check_pts_in_occ_voxel forward func
   @param: xyz, torch float tensor of (B, 3)
   @param: bitfield, (N_grid, N_grid, N_grid), bool tensor indicating each voxel's occupancy
   @param: range, torch float tensor of (3, 2), range of xyz boundary
   @param: n_grid, uint8_t resolution
   @return: output, torch bool tensor of (B,)
*/
torch::Tensor check_pts_in_occ_voxel(
    const torch::Tensor xyz,
    const torch::Tensor bitfield,
    const torch::Tensor range,
    const uint32_t n_grid){
    // checking
    CHECK_INPUT(xyz)
    CHECK_IS_FLOATING(xyz)
    CHECK_INPUT(bitfield)
    CHECK_IS_BOOL(bitfield)
    CHECK_INPUT(range)
    CHECK_IS_FLOATING(range)

    if (xyz.size(1) != 3) {
        throw std::runtime_error{"Input tensor must be (B, 3)."};
    }

    if (range.size(0) != 3 || range.size(1) != 2) {
        throw std::runtime_error{"xyz range should be in (3, 2)."};
    }

    if (bitfield.size(0) != n_grid || bitfield.size(1) != n_grid || bitfield.size(2) != n_grid) {
        throw std::runtime_error{"bitfield should be in (n_grid, n_grid, n_grid)."};
    }

    // call actual cuda function
    return check_pts_in_occ_voxel_cuda(xyz, bitfield, range, n_grid);
}


// define the real cuda function to be called by c++ wrapper.
std::vector<torch::Tensor> aabb_intersection_cuda(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor aabb_range,
    const float eps);


/* c++ wrapper of aabb intersection func
   @param: rays_o, ray origin, (N_rays, 3)
   @param: rays_d, ray direction, assume normalized, (N_rays, 3)
   @param: aabb_range, bbox range of volume, (N_v, 3, 2) of xyz_min/max of each volume
   @param: eps, error threshold
   @return: near, near intersection zvals. (N_rays, N_v)
   @return: far, far intersection zvals. (N_rays, N_v)
   @return: pts, intersection pts with the volume. (N_rays, N_v, 2, 3)
   @return: mask, (N_rays, N_v), show whether each ray has intersection with the volume, BoolTensor
*/
std::vector<torch::Tensor> aabb_intersection(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor aabb_range,
    const float eps){
    // checking
    CHECK_INPUT(rays_o)
    CHECK_IS_FLOATING(rays_o)
    CHECK_INPUT(rays_d)
    CHECK_IS_FLOATING(rays_d)
    CHECK_INPUT(aabb_range)
    CHECK_IS_FLOATING(aabb_range)

    if (rays_o.size(1) != 3 || rays_d.size(1) != 3) {
        throw std::runtime_error{"Input rays tensor must be (B, 3)."};
    }

    if (aabb_range.size(1) != 3 || aabb_range.size(2) != 2) {
        throw std::runtime_error{"xyz range should be in (B, 3, 2)."};
    }

    // call actual cuda function
    return aabb_intersection_cuda(rays_o, rays_d, aabb_range, eps);
}


// define the real cuda function to be called by c++ wrapper.
std::vector<torch::Tensor> sparse_volume_sampling_cuda(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor near,
    const torch::Tensor far,
    const int N_pts,
    const float dt,
    const torch::Tensor aabb_range,
    const int n_grid,
    const torch::Tensor bitfield,
    const float near_distance,
    const bool perturb);


/* c++ wrapper of sparse_volume_sampling forward func
   @param: rays_o, ray origin, (N_rays, 3)
   @param: rays_d, ray direction, assume normalized, (N_rays, 3)
   @param: near, near intersection zvals. (N_rays, 1)
   @param: far, far intersection zvals. (N_rays, 1)
   @param: N_pts, max num of sampling pts on each ray.
   @param: dt, fix step length
   @param: near_distance, near distance for sampling. By default 0.0.
   @param: perturb, whether to perturb the first zval, use in training only
   @return: pts, (N_rays, N_pts, 3), sampled points on each rays.
   @return: mask, (N_rays, N_pts), show whether each ray has intersection with the volume, BoolTensor
*/
std::vector<torch::Tensor> sparse_volume_sampling(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor near,
    const torch::Tensor far,
    const int N_pts,
    const float dt,
    const torch::Tensor aabb_range,
    const int n_grid,
    const torch::Tensor bitfield,
    const float near_distance,
    const bool perturb){
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
    CHECK_IS_BOOL(bitfield)

    if (rays_o.size(1) != 3 || rays_d.size(1) != 3) {
        throw std::runtime_error{"Input rays tensor must be (B, 3)."};
    }

    if (near.size(1) != 1 || far.size(1) != 1) {
        throw std::runtime_error{"Input near/far tensor must be (B, 1)."};
    }

    if (aabb_range.size(0) != 3 || aabb_range.size(1) != 2) {
        throw std::runtime_error{"xyz range should be in (3, 2)."};
    }

    if (bitfield.size(0) != n_grid || bitfield.size(1) != n_grid || bitfield.size(2) != n_grid) {
        throw std::runtime_error{"bitfield should be in (n_grid, n_grid, n_grid)."};
    }

    // call actual cuda function
    return sparse_volume_sampling_cuda(
        rays_o, rays_d, near, far, N_pts, dt, aabb_range, n_grid, bitfield, near_distance, perturb
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("check_pts_in_occ_voxel", &check_pts_in_occ_voxel, "check pts in occ voxel (CUDA)");
    m.def("aabb_intersection", &aabb_intersection, "aabb intersection (CUDA)");
    m.def("sparse_volume_sampling", &sparse_volume_sampling, "sparse volume sampling (CUDA)");
}
