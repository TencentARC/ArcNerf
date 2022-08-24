// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// volume related func in cuda


#include <torch/extension.h>

#include "helper.h"
#include "utils.h"


// The real cuda kernel
template <typename scalar_t>
__global__ void check_pts_in_occ_voxel_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz,
    const torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> bitfield,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> range,
    const uint32_t n_grid,
    torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> output) {

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= xyz.size(0)) return;

    // grid size
    scalar_t voxel_size[3];
    # pragma unroll
    for (uint32_t i=0; i<3; i++) {
        voxel_size[i] = ((range[i][1] - range[i][0]) / (scalar_t)(n_grid));
    }

    // voxel index
    int32_t voxel_idx[3];
    # pragma unroll
    for (uint32_t i=0; i<3; i++) {
        voxel_idx[i] = (int32_t)((xyz[n][i] - range[i][0]) / voxel_size[i]);
    }

    // valid for this pts
    # pragma unroll
    for (uint32_t i=0; i<3; i++) {
        if (voxel_idx[i] < 0 || voxel_idx[i] >= n_grid)
            return;  // not in the volume
    }

    const bool occ = bitfield[voxel_idx[0]][voxel_idx[1]][voxel_idx[2]];
    if (occ) {
        output[n] = true;
    }

    return;
}

/* CUDA instantiate func for hashgrid_encode forward
 @param: xyz, torch float tensor of (B, 3)
   @param: (N_grid, N_grid, N_grid), bool tensor indicating each voxel's occupancy
   @param: range, torch float tensor of (3, 2), range of xyz boundary
   @param: n_grid, uint8_t resolution
   @return: output, torch bool tensor of (B,)
*/
torch::Tensor check_pts_in_occ_voxel_cuda(
    const torch::Tensor xyz,
    const torch::Tensor bitfield,
    const torch::Tensor range,
    const uint32_t n_grid) {

    const uint32_t B = xyz.size(0);  // B

    const uint32_t threads = 512;
    const uint32_t blocks = div_round_up(B, threads);

    // Init the output tensor
    torch::Tensor output = torch::zeros({B,}, torch::kBool).to(xyz.device());  // (B,)

    // instantiate the real executable kernel
    AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "check_pts_in_occ_voxel_cuda",
    ([&] {
        check_pts_in_occ_voxel_cuda_kernel<scalar_t><<<blocks, threads>>>(
            xyz.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            bitfield.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
            range.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            n_grid,
            output.packed_accessor32<bool, 1, torch::RestrictPtrTraits>()
        );
    }));

    return output;

}


// The real cuda kernel
template <typename scalar_t>
__global__ void aabb_intersection_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rays_o,  //(N_rays, 3)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rays_d,  //(N_rays, 3)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> aabb_range,  //(N_v, 3, 2)
    const float eps,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> near,  //(N_rays, N_v)
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> far,  //(N_rays, N_v)
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> pts,  //(N_rays, N_v, 2, 3)
    torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> mask) {  //(N_rays, N_v)

    const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;  // ray id
    const uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;  // volume id

    if (n < rays_o.size(0) && v < aabb_range.size(0)) {
        // dim 0
         scalar_t abs_d = abs(rays_d[n][0]);
        bool mask_axis = (abs_d < eps);
        bool mask_axis_out = (rays_o[n][0] < aabb_range[v][0][0]) || (rays_o[n][0] > aabb_range[v][0][1]);
        if (mask_axis && mask_axis_out) mask[n][v] = false;

        scalar_t t1 = (aabb_range[v][0][0] - rays_o[n][0]) / rays_d[n][0];
        scalar_t t2 = (aabb_range[v][0][1] - rays_o[n][0]) / rays_d[n][0];
        if (t1 > t2) host_device_swap(t1, t2);  // t1 < t2
        if (mask[n][v] && t1 > near[n][v]) near[n][v] = t1;
        if (mask[n][v] && t2 < far[n][v]) far[n][v] = t2;
        if (near[n][v] > far[n][v]) mask[n][v] = false;

        // dim 1
        abs_d = abs(rays_d[n][1]);
        mask_axis = (abs_d < eps);
        mask_axis_out = (rays_o[n][1] < aabb_range[v][1][0]) || (rays_o[n][1] > aabb_range[v][1][1]);
        if (mask_axis && mask_axis_out) mask[n][v] = false;

        t1 = (aabb_range[v][1][0] - rays_o[n][1]) / rays_d[n][1];
        t2 = (aabb_range[v][1][1] - rays_o[n][1]) / rays_d[n][1];
        if (t1 > t2) host_device_swap(t1, t2);  // t1 < t2
        if (mask[n][v] && t1 > near[n][v]) near[n][v] = t1;
        if (mask[n][v] && t2 < far[n][v]) far[n][v] = t2;
        if (near[n][v] > far[n][v]) mask[n][v] = false;

        // dim 2
        abs_d = abs(rays_d[n][2]);
        mask_axis = (abs_d < eps);
        mask_axis_out = (rays_o[n][1] < aabb_range[v][2][0]) || (rays_o[n][2] > aabb_range[v][2][1]);
        if (mask_axis && mask_axis_out) mask[n][v] = false;

        t1 = (aabb_range[v][2][0] - rays_o[n][2]) / rays_d[n][2];
        t2 = (aabb_range[v][2][1] - rays_o[n][2]) / rays_d[n][2];
        if (t1 > t2) host_device_swap(t1, t2);  // t1 < t2
        if (mask[n][v] && t1 > near[n][v]) near[n][v] = t1;
        if (mask[n][v] && t2 < far[n][v]) far[n][v] = t2;
        if (near[n][v] > far[n][v]) mask[n][v] = false;

        // post process
        near[n][v] = max(0.0, near[n][v]);
        far[n][v] = max(0.0, far[n][v]);
        if (mask[n][v] == false) {
            near[n][v] = 0.0;
            far[n][v] = 0.0;
        } else {
            near[n][v] += eps;
            far[n][v] += eps;
        }

        // get ray pts
        pts[n][v][0][0] = rays_o[n][0] + near[n][v] * rays_d[n][0];
        pts[n][v][0][1] = rays_o[n][1] + near[n][v] * rays_d[n][1];
        pts[n][v][0][2] = rays_o[n][2] + near[n][v] * rays_d[n][2];
        pts[n][v][1][0] = rays_o[n][0] + far[n][v] * rays_d[n][0];
        pts[n][v][1][1] = rays_o[n][1] + far[n][v] * rays_d[n][1];
        pts[n][v][1][2] = rays_o[n][2] + far[n][v] * rays_d[n][2];
    }

    return;
}

/* CUDA instantiate of aabb intersection forward func
   @param: ray origin, (N_rays, 3)
   @param: ray direction, assume normalized, (N_rays, 3)
   @param: bbox range of volume, (N_v, 3, 2) of xyz_min/max of each volume
   @return: near, near intersection zvals. (N_rays, N_v)
   @return: far, far intersection zvals. (N_rays, N_v)
   @return: mask, (N_rays, N_v), show whether each ray has intersection with the volume, BoolTensor
*/
std::vector<torch::Tensor> aabb_intersection_cuda(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor aabb_range,
    const float eps) {

    const uint32_t N_rays = rays_o.size(0);  // N_rays
    const uint32_t N_v = aabb_range.size(0); // N_v

    const uint32_t threads_per_row = 512;
    const uint32_t threads_per_col = 1;  // commonly use one volume to check
    const dim3 threads(threads_per_row, threads_per_col);
    const dim3 blocks(div_round_up(N_rays, threads_per_row), div_round_up(N_v, threads_per_col));

    // Init the output tensor
    auto dtype = rays_o.dtype();
    auto device = rays_o.device();
    torch::Tensor near = torch::zeros({N_rays, N_v}, dtype).to(device);  // (N_rays, N_v)
    torch::Tensor far = torch::ones({N_rays, N_v}, dtype).to(device) * 10000.0f;  // (N_rays, N_v)
    torch::Tensor pts = torch::zeros({N_rays, N_v, 2, 3}, dtype).to(device);  // (N_rays, N_v, 2, 3)
    torch::Tensor mask = torch::ones({N_rays, N_v}, torch::kBool).to(device);  // (N_rays, N_v)

    // instantiate the real executable kernel
    AT_DISPATCH_FLOATING_TYPES(rays_o.scalar_type(), "aabb_intersection_cuda",
    ([&] {
        aabb_intersection_cuda_kernel<scalar_t><<<blocks, threads>>>(
            rays_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            aabb_range.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            eps,
            near.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            far.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            pts.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            mask.packed_accessor32<bool, 2, torch::RestrictPtrTraits>()
        );
    }));

    return {near, far, pts, mask};
}
