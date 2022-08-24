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
   @param: (N_grid, N_grid, N_grid), bool tensor indicating each voxel's occupancy
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


/* c++ wrapper of aabb intersection forward func
   @param: ray origin, (N_rays, 3)
   @param: ray direction, assume normalized, (N_rays, 3)
   @param: bbox range of volume, (N_v, 3, 2) of xyz_min/max of each volume
   @return: near, near intersection zvals. (N_rays, N_v)
   @return: far, far intersection zvals. (N_rays, N_v)
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("check_pts_in_occ_voxel", &check_pts_in_occ_voxel, "check pts in occ voxel (CUDA)");
    m.def("aabb_intersection", &aabb_intersection, "aabb intersection (CUDA)");
}
