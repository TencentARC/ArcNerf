// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// densegrid related func


#include <torch/torch.h>

#include "utils.h"


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
    const int n_grid,
    const int n_cascades);

void update_bitfield(
    const torch::Tensor density_grid,
    const float density_grid_mean,
    torch::Tensor density_grid_bitfield,
    const float thres,
    const int n_grid,
    const int n_cascades) {

    // checking
    CHECK_INPUT(density_grid)
    CHECK_IS_FLOATING(density_grid)
    CHECK_INPUT(density_grid_bitfield)

    return update_bitfield_cuda(density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid, n_cascades);
}

// -------------------------------------------------- ------------------------------------ //


void splat_grid_samples_cuda(
    const torch::Tensor density,
    const torch::Tensor density_grid_indices,
    const int n_density_grid_samples,
    const float dt,
    torch::Tensor density_grid_tmp);

void splat_grid_samples(
    const torch::Tensor density,
    const torch::Tensor density_grid_indices,
    const int n_density_grid_samples,
    const float dt,
    torch::Tensor density_grid_tmp) {

    // checking
    CHECK_INPUT(density)
    CHECK_IS_FLOATING(density)
    CHECK_INPUT(density_grid_indices)
    CHECK_IS_INT(density_grid_indices)
    CHECK_INPUT(density_grid_tmp)
    CHECK_IS_FLOATING(density_grid_tmp)

    return splat_grid_samples_cuda(
        density, density_grid_indices, n_density_grid_samples, dt, density_grid_tmp);

}

// -------------------------------------------------- ------------------------------------ //


void generate_grid_samples_cuda(
    const torch::Tensor density_grid,
    const int density_grid_ema_step,
    const int n_elements,
    const int max_cascade,
    const int n_grid,
    const float thresh,
    torch::Tensor density_grid_positions_uniform,
    torch::Tensor density_grid_indices_uniform);

void generate_grid_samples(
    const torch::Tensor density_grid,
    const int density_grid_ema_step,
    const int n_elements,
    const int max_cascade,
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
        density_grid, density_grid_ema_step, n_elements, max_cascade, n_grid, thresh,
        density_grid_positions_uniform, density_grid_indices_uniform);
}


// -------------------------------------------------- ------------------------------------ //


void count_bitfield_cuda(const torch::Tensor density_grid_bitfield, torch::Tensor counter, const int n_grid, const int level);

void count_bitfield(const torch::Tensor density_grid_bitfield, torch::Tensor counter, const int n_grid, const int level) {
    return count_bitfield_cuda(density_grid_bitfield, counter, n_grid, level);
}

// -------------------------------------------------- ------------------------------------ //


void get_occ_pc_cuda(
    const torch::Tensor density_grid_bitfield,
    torch::Tensor pc,
    torch::Tensor counter,
    const int n_grid);

void get_occ_pc(
    const torch::Tensor density_grid_bitfield,
    torch::Tensor pc,
    torch::Tensor counter,
    const int n_grid) {

    return get_occ_pc_cuda(density_grid_bitfield, pc, counter, n_grid);
}

// -------------------------------------------------- ------------------------------------ //


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ema_grid_samples_nerf", &ema_grid_samples_nerf, "ema_grid_samples_nerf (CUDA)");
    m.def("update_bitfield", &update_bitfield, "update_bitfield (CUDA)");
    m.def("splat_grid_samples", &splat_grid_samples, "splat_grid_samples (CUDA)");
    m.def("generate_grid_samples", &generate_grid_samples, "generate_grid_samples (CUDA)");
    m.def("count_bitfield", &count_bitfield, "count_bitfield (CUDA)");
    m.def("get_occ_pc", &get_occ_pc, "get_occ_pc (CUDA)");
}
