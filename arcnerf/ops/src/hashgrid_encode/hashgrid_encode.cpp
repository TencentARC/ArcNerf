// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// multi-res grid vertices hash embeddings

#include <torch/torch.h>

#include "utils.h"

// define the real cuda function to be called by c++ wrapper.
torch::Tensor hashgrid_encode_forward_cuda(
    const torch::Tensor xyz,
    const torch::Tensor embeddings,
    const uint32_t L,
    const uint32_t F,
    const std::vector<uint32_t> offsets,
    const std::vector<uint32_t> resolutions,
    const torch::Tensor min_xyz,
    const torch::Tensor max_xyz,
    const bool cal_grad,
    torch::Tensor weights,
    torch::Tensor hash_idx,
    torch::Tensor valid,
    torch::Tensor dw_dxyz);


/* c++ wrapper of hashgrid_encode forward func
   py: hashgrid_encode_forward(xyz, embeddings, grad_xyz, grad_embeddings, n_levels, n_feat_per_entry,
                               offsets, resolutions, min_xyz, max_xyz)
   @param: xyz, torch float tensor of (B, D)
   @param: embeddings, torch float tensor of (n_total_embed, F)
   @param: L, num of levels of embedding(L), by default 16
   @param: F, num of feat for each entry in hashmap(F), by default 2
   @param: offsets, list of (L+1, ), offset of each level, len is L+1
   @param: resolutions, list of (L, ), resolution at each level, len is L
   @param: min_xyz, torch float tensor of (D, ), the min_xyz position of the grid
   @param: max_xyz, torch float tensor of (D, ), the max_xyz position of the grid
   @param: cal_grad, bool value decide whether to cal grad
   @param: weights, torch float tensor of (B, L, 1<<D), the contributed weights in each level on each grid_pts
   @param: hash_idx, torch long tensor of (B, L, 1<<D), the hash index of pts in each level on each grid_pts
   @param: valid, torch bool tensor of (B,), whether the pts is in grid
   @param: dw_dxyz, torch float tensor of (B, L, 1<<D, D), save the grad from w_xyz to xyz
   @return: output, torch float tensor of (B, L, F)
*/
torch::Tensor hashgrid_encode_forward(
    const torch::Tensor xyz,
    const torch::Tensor embeddings,
    const uint32_t L,
    const uint32_t F,
    const std::vector<uint32_t> offsets,
    const std::vector<uint32_t> resolutions,
    const torch::Tensor min_xyz,
    const torch::Tensor max_xyz,
    const bool cal_grad,
    torch::Tensor weights,
    torch::Tensor hash_idx,
    torch::Tensor valid,
    torch::Tensor dw_dxyz) {
    // checking
    CHECK_INPUT(xyz)
    CHECK_IS_FLOATING(xyz)
    CHECK_INPUT(embeddings)
    CHECK_IS_FLOATING(embeddings)
    CHECK_INPUT(min_xyz)
    CHECK_IS_FLOATING(min_xyz)
    CHECK_INPUT(max_xyz)
    CHECK_IS_FLOATING(max_xyz)
    CHECK_INPUT(weights)
    CHECK_IS_FLOATING(weights)
    CHECK_INPUT(hash_idx)
    CHECK_IS_LONG(hash_idx)
    CHECK_INPUT(valid)
    CHECK_IS_BOOL(valid)
    CHECK_INPUT(dw_dxyz)
    CHECK_IS_FLOATING(dw_dxyz)


    if (offsets.size() != L + 1) {
        throw std::runtime_error{"Offset length must be L+1."};
    }

    int n_total_embed = offsets[L];
    if (embeddings.size(0) != n_total_embed || embeddings.size(1) != F) {
        throw std::runtime_error{"embeddings tensor must be (n_total_embed, F)."};
    }

    if (resolutions.size() != L) {
        throw std::runtime_error{"Resolutions length must be L."};
    }

    if (min_xyz.size(0) != (uint32_t)xyz.size(1) || max_xyz.size(0) != (uint32_t)xyz.size(1)) {
        throw std::runtime_error{"xyz boundary length must be same as xyz."};
    }

    // call actual cuda function
    return hashgrid_encode_forward_cuda(xyz, embeddings, L, F, offsets, resolutions, min_xyz, max_xyz, cal_grad,
                                        weights, hash_idx, valid, dw_dxyz);
}

// define the real cuda function to be called by c++ wrapper.
std::vector<torch::Tensor> hashgrid_encode_backward_cuda(
    const torch::Tensor grad_out,
    const torch::Tensor xyz,
    const torch::Tensor embeddings,
    const torch::Tensor weights,
    const torch::Tensor hash_idx,
    const torch::Tensor valid,
    const torch::Tensor dw_dxyz);


/* c++ wrapper of hashgrid_encode backward func
   py: hashgrid_encode_bacward(grad, grad_xyz, grad_embeddings)
   @param: grad_out, torch float tensor of (B, L, F), final grad
   @param: xyz, torch float tensor of (B, D)
   @param: embeddings, torch float tensor of (n_total_embed, F)
   @param: weights, torch float tensor of (B, L, 1<<D), the contributed weights in each level on each grid_pts
   @param: hash_idx, torch long tensor of (B, L, 1<<D), the hash index of pts in each level on each grid_pts
   @param: valid, torch bool tensor of (B,), whether the pts is in grid
   @return: list of output, first is grad_xyz (B, D), second is grad_embeddings (n_total_embed, F)
*/
std::vector<torch::Tensor> hashgrid_encode_backward(
    const torch::Tensor grad_out,
    const torch::Tensor xyz,
    const torch::Tensor embeddings,
    const torch::Tensor weights,
    const torch::Tensor hash_idx,
    const torch::Tensor valid,
    const torch::Tensor dw_dxyz) {
    // checking
    CHECK_INPUT(grad_out)
    CHECK_IS_FLOATING(grad_out)
    CHECK_INPUT(xyz)
    CHECK_IS_FLOATING(xyz)
    CHECK_INPUT(embeddings)
    CHECK_IS_FLOATING(embeddings)
    CHECK_INPUT(weights)
    CHECK_IS_FLOATING(weights)
    CHECK_INPUT(hash_idx)
    CHECK_IS_LONG(hash_idx)
    CHECK_INPUT(valid)
    CHECK_IS_BOOL(valid)
    CHECK_INPUT(dw_dxyz)
    CHECK_IS_FLOATING(dw_dxyz)

    // call actual cuda function
    return hashgrid_encode_backward_cuda(grad_out, xyz, embeddings, weights, hash_idx, valid, dw_dxyz);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hashgrid_encode_forward", &hashgrid_encode_forward, "hashgrid encode forward (CUDA)");
    m.def("hashgrid_encode_backward", &hashgrid_encode_backward, "hashgrid encode backward (CUDA)");
}
