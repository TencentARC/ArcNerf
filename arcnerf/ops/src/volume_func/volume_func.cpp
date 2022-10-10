// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// volume related func in cuda


#include <torch/torch.h>

#include "utils.h"


// define the real cuda function to be called by c++ wrapper.
void check_pts_in_occ_voxel_cuda(
    const torch::Tensor xyz,
    const torch::Tensor bitfield,
    const torch::Tensor aabb_range,
    const int n_grid,
    torch::Tensor output);


/* c++ wrapper of check_pts_in_occ_voxel forward func
   @param: xyz, torch float tensor of (B, 3)
   @param: bitfield, (N_grid, N_grid, N_grid), bool tensor indicating each voxel's occupancy
   @param: range, torch float tensor of (2, 3), range of xyz boundary
   @param: n_grid, resolution
   @return: output, torch bool tensor of (B,)
*/
void check_pts_in_occ_voxel(
    const torch::Tensor xyz,
    const torch::Tensor bitfield,
    const torch::Tensor aabb_range,
    const int n_grid,
    torch::Tensor output){
    // checking
    CHECK_INPUT(xyz)
    CHECK_IS_FLOATING(xyz)
    CHECK_INPUT(bitfield)
    CHECK_IS_BOOL(bitfield)
    CHECK_INPUT(aabb_range)
    CHECK_IS_FLOATING(aabb_range)
    CHECK_INPUT(output)
    CHECK_IS_BOOL(output)

    if (xyz.size(1) != 3) {
        throw std::runtime_error{"Input tensor must be (B, 3)."};
    }

    if (aabb_range.size(0) != 2 || aabb_range.size(1) != 3) {
        throw std::runtime_error{"xyz range should be in (2, 3)."};
    }

    if (bitfield.size(0) != n_grid || bitfield.size(1) != n_grid || bitfield.size(2) != n_grid) {
        throw std::runtime_error{"bitfield should be in (n_grid, n_grid, n_grid)."};
    }

   if (xyz.size(0) != output.size(0)) {
        throw std::runtime_error{"xyz must have same n_pts as output."};
    }

    // call actual cuda function
    return check_pts_in_occ_voxel_cuda(xyz, bitfield, aabb_range, n_grid, output);
}


// ------------------------------------------------------------------------------------------------ //


// define the real cuda function to be called by c++ wrapper.
void aabb_intersection_cuda(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor aabb_range,
    torch::Tensor near,
    torch::Tensor far,
    torch::Tensor pts,
    torch::Tensor mask);


/* c++ wrapper of aabb intersection func
   @param: rays_o, ray origin, (N_rays, 3)
   @param: rays_d, ray direction, assume normalized, (N_rays, 3)
   @param: aabb_range, bbox range of volume, (N_v, 2, 3) of xyz_min/max of each volume
   @return: near, near intersection zvals. (N_rays, N_v)
   @return: far, far intersection zvals. (N_rays, N_v)
   @return: pts, intersection pts with the volume. (N_rays, N_v, 2, 3)
   @return: mask, (N_rays, N_v), show whether each ray has intersection with the volume, BoolTensor
*/
void aabb_intersection(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor aabb_range,
    torch::Tensor near,
    torch::Tensor far,
    torch::Tensor pts,
    torch::Tensor mask){
    // checking
    CHECK_INPUT(rays_o)
    CHECK_IS_FLOATING(rays_o)
    CHECK_INPUT(rays_d)
    CHECK_IS_FLOATING(rays_d)
    CHECK_INPUT(aabb_range)
    CHECK_IS_FLOATING(aabb_range)
    CHECK_INPUT(near)
    CHECK_IS_FLOATING(near)
    CHECK_INPUT(far)
    CHECK_IS_FLOATING(far)
    CHECK_INPUT(pts)
    CHECK_IS_FLOATING(pts)
    CHECK_INPUT(mask)
    CHECK_IS_BOOL(mask)

    if (rays_o.size(1) != 3 || rays_d.size(1) != 3) {
        throw std::runtime_error{"Input rays tensor must be (B, 3)."};
    }

    if (aabb_range.size(1) != 2 || aabb_range.size(2) != 3) {
        throw std::runtime_error{"xyz range should be in (B, 2, 3)."};
    }

    if (near.size(0) != rays_o.size(0) || near.size(1) != aabb_range.size(0)) {
        throw std::runtime_error{"near should be in (B, V)."};
    }

    if (far.size(0) != rays_o.size(0) || far.size(1) != aabb_range.size(0)) {
        throw std::runtime_error{"far should be in (B, V)."};
    }

    if (pts.size(0) != rays_o.size(0) || pts.size(1) != aabb_range.size(0) || pts.size(2) != 2 || pts.size(3) != 3) {
        throw std::runtime_error{"pts should be in (B, V, 2, 3)."};
    }

    if (mask.size(0) != rays_o.size(0) || mask.size(1) != aabb_range.size(0)) {
        throw std::runtime_error{"mask should be in (B, V)."};
    }

    // call actual cuda function
    return aabb_intersection_cuda(rays_o, rays_d, aabb_range, near, far, pts, mask);
}

// ------------------------------------------------------------------------------------------------ //


// define the real cuda function to be called by c++ wrapper.
void sparse_volume_sampling_cuda(
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
   @param: near_distance, near distance for sampling. By default 0.0.
   @return: zvals, (N_rays, N_pts), sampled points zvals on each rays.
   @return: mask, (N_rays, N_pts), show whether each ray has intersection with the volume, BoolTensor
*/
void sparse_volume_sampling(
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
    CHECK_IS_BOOL(bitfield)
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

    if (bitfield.size(0) != n_grid || bitfield.size(1) != n_grid || bitfield.size(2) != n_grid) {
        throw std::runtime_error{"bitfield should be in (n_grid, n_grid, n_grid)."};
    }

    if (zvals.size(0) != rays_o.size(0) || zvals.size(1) != n_pts) {
        throw std::runtime_error{"zval should be in (n_rays, n_pts)."};
    }

    if (mask.size(0) != rays_o.size(0) || mask.size(1) != n_pts) {
        throw std::runtime_error{"mask should be in (n_rays, n_pts)."};
    }

    // call actual cuda function
    return sparse_volume_sampling_cuda(
        rays_o, rays_d, near, far, n_pts, dt, aabb_range, n_grid, bitfield, near_distance,
        zvals, mask
    );
}

// ------------------------------------------------------------------------------------------------ //


// define the real cuda function to be called by c++ wrapper.
void tensor_reduce_max_cuda(
    const torch::Tensor full_tensor,
    const torch::Tensor group_idx,
    const int n_group,
    torch::Tensor uni_tensor);

/* c++ wrapper of tensor_reduce_max forward func
   @param: full_tensor: full value tensor, (N, )
   @param: group_idx: index of each row (N, )
   @param: n_group: num of group (N_uni)
   @return: uni_tensor: (N_uni. ) maximum of each unique group
*/
void tensor_reduce_max(
    const torch::Tensor full_tensor,
    const torch::Tensor group_idx,
    const int n_group,
    torch::Tensor uni_tensor){
    // checking
    CHECK_INPUT(full_tensor)
    CHECK_IS_FLOATING(full_tensor)
    CHECK_INPUT(group_idx)
    CHECK_IS_LONG(group_idx)
    CHECK_INPUT(uni_tensor)
    CHECK_IS_FLOATING(uni_tensor)

    if (full_tensor.size(0) != group_idx.size(0)) {
        throw std::runtime_error{"Full tensor same as idx."};
    }

    if (uni_tensor.size(0) != n_group) {
        throw std::runtime_error{"uni tensor size of (n_group,)."};
    }

    // call actual cuda function
    return tensor_reduce_max_cuda(full_tensor, group_idx, n_group, uni_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("check_pts_in_occ_voxel", &check_pts_in_occ_voxel, "check pts in occ voxel (CUDA)");
    m.def("aabb_intersection", &aabb_intersection, "aabb intersection (CUDA)");
    m.def("sparse_volume_sampling", &sparse_volume_sampling, "sparse volume sampling (CUDA)");
    m.def("tensor_reduce_max", &tensor_reduce_max, "tensor reduce max (CUDA)");
}
