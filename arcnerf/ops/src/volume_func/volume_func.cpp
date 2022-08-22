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
   py: check_pts_in_occ_voxel(xyz, degree)
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("check_pts_in_occ_voxel", &check_pts_in_occ_voxel, "check pts in occ voxel (CUDA)");
}
