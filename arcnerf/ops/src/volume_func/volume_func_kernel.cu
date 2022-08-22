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
    AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "hashgrid_encode_forward_cuda",
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
