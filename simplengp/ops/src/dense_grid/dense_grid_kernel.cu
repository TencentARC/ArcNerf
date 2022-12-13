// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// densegrid related func


#include <torch/extension.h>

#include "common.h"
extern pcg32 rng;


// -------------------------------------------------- ------------------------------------ //


__global__ void ema_grid_samples_nerf_cuda_kernel(
    const uint32_t n_elements,
    float decay,
    float *__restrict__ grid_out,
    const float *__restrict__ grid_in)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    float importance = grid_in[i];
    float prev_val = grid_out[i];

    float val = (prev_val < 0.f) ? prev_val : fmaxf(prev_val * decay, importance);
    grid_out[i] = val;
}

void ema_grid_samples_nerf_cuda(
    const torch::Tensor density_grid_tmp,
    int n_elements,
    float decay,
    torch::Tensor density_grid)
{
    cudaStream_t stream=0;

    // input
    uint32_t u_n_elements = (uint32_t)n_elements;
    float* density_grid_tmp_p = (float*)density_grid_tmp.data_ptr();
    // output
    float* density_grid_p = (float*)density_grid.data_ptr();

    linear_kernel(ema_grid_samples_nerf_cuda_kernel, 0, stream,
        u_n_elements, decay, density_grid_p, density_grid_tmp_p);

    cudaDeviceSynchronize();
}



// -------------------------------------------------- ------------------------------------ //


__global__ void grid_to_bitfield(
    const uint32_t n_elements,
    const float *__restrict__ grid,
    uint8_t *__restrict__ grid_bitfield,
    const float mean_density,
    const float opa_thres)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    uint8_t bits = 0;

    float thresh = opa_thres < mean_density ? opa_thres : mean_density;

    #pragma unroll
    for (uint8_t j = 0; j < 8; ++j)
    {
        bits |= grid[i * 8 + j] > thresh ? ((uint8_t)1 << j) : 0;
    }

    grid_bitfield[i] = bits;
}

__global__ void bitfield_max_pool(
    const uint32_t n_elements,
    const uint8_t *__restrict__ prev_level,
    uint8_t *__restrict__ next_level,
    const uint32_t n_grid)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    uint8_t bits = 0;

    #pragma unroll
    for (uint8_t j = 0; j < 8; ++j)
    {
        bits |= prev_level[i * 8 + j] > 0 ? ((uint8_t)1 << j) : 0;
    }

    uint32_t x = morton3D_invert(i >> 0) + n_grid / 8;
    uint32_t y = morton3D_invert(i >> 1) + n_grid / 8;
    uint32_t z = morton3D_invert(i >> 2) + n_grid / 8;

    next_level[morton3D(x, y, z)] |= bits;

}

void update_bitfield_cuda(
    const torch::Tensor density_grid,
    const float density_grid_mean,
    torch::Tensor density_grid_bitfield,
    const float thres,
    const int n_grid,
    const int n_cascades) {

    cudaStream_t stream=0;
    // input
    float* density_grid_p = (float*)density_grid.data_ptr();
    // output
    uint8_t* density_grid_bitfield_p = (uint8_t*)density_grid_bitfield.data_ptr();

    const uint32_t u_n_grid = (uint32_t)n_grid;
    const uint32_t u_n_cascades = (uint32_t)n_cascades;
    const uint32_t n_elements = u_n_grid * u_n_grid * u_n_grid;

    linear_kernel(grid_to_bitfield, 0, stream, n_elements / 8 * u_n_cascades,
        density_grid_p, density_grid_bitfield_p, density_grid_mean, thres);

    for (uint32_t level = 1; level < u_n_cascades; ++level)
        {{
        linear_kernel(bitfield_max_pool, 0, stream, n_elements / 64,
            density_grid_bitfield_p + grid_mip_offset(level-1, n_grid)/8,
            density_grid_bitfield_p + grid_mip_offset(level, n_grid)/8,
            u_n_grid);
        }}

    cudaDeviceSynchronize();
}


// -------------------------------------------------- ------------------------------------ //


template <typename T>
__global__ void splat_grid_samples_cuda_kernel(
    const uint32_t n_elements,
    const uint32_t *__restrict__ indices,
    const T *density_output,
    const float dt,
    float *__restrict__ grid_out)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    uint32_t local_idx = indices[i];

    float density = density_output[i];  // already applied activation in model
    float optical_thickness = density * dt;

    // Positive floats are monotonically ordered when their bit pattern is interpretes as uint.
    // uint atomicMax is thus perfectly acceptable.
    atomicMax((uint32_t *)&grid_out[local_idx], __float_as_uint(optical_thickness));
}

void splat_grid_samples_cuda(
    const torch::Tensor density,
    const torch::Tensor density_grid_indices,
    const int n_density_grid_samples,
    const float dt,
    torch::Tensor density_grid_tmp) {

    cudaStream_t stream=0;
    // input
    uint32_t u_n_density_grid_samples = n_density_grid_samples;
    uint32_t* density_grid_indices_p = (uint32_t*)density_grid_indices.data_ptr();
    float* density_p = (float*)density.data_ptr();
    // output
    float* density_grid_tmp_p = (float*)density_grid_tmp.data_ptr();

    linear_kernel(splat_grid_samples_cuda_kernel<float>, 0, stream,
        u_n_density_grid_samples, density_grid_indices_p, density_p, dt, density_grid_tmp_p);

    cudaDeviceSynchronize();
}


// -------------------------------------------------- ------------------------------------ //


__global__ void generate_grid_samples_cuda_kernel(
    const uint32_t n_elements,
    default_rng_t rng,
    const uint32_t step,
    const float *__restrict__ grid_in,
    Vector3f *__restrict__ out,
    uint32_t *__restrict__ indices,
    uint32_t n_cascades,
    uint32_t n_grid,
    const float thresh)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n_elements)
        return;

    // 1 random number to select the level, 3 to select the position.
    rng.advance(i * 4);
    uint32_t level = (uint32_t)(random_val(rng) * n_cascades) % n_cascades;
    uint32_t n_grid_per_level = n_grid * n_grid * n_grid;

    // Select grid cell that has density
    uint32_t idx;
    // uint32_t step=*step_p; # use input param
    for (uint32_t j = 0; j < 10; ++j)
    {
        idx = ((i + step * n_elements) * 56924617 + j * 19349663 + 96925573) % n_grid_per_level;
        idx += level * n_grid_per_level;
        if (grid_in[idx] > thresh)
        {
            break;
        }
    }

    // Random position within that cellq
    uint32_t pos_idx = idx % n_grid_per_level;
    uint32_t x = morton3D_invert(pos_idx >> 0);
    uint32_t y = morton3D_invert(pos_idx >> 1);
    uint32_t z = morton3D_invert(pos_idx >> 2);

    // add noise inside voxel
    Eigen::Vector3f pos = ((Eigen::Vector3f{(float)x, (float)y, (float)z} + random_val_3d(rng)) / n_grid - Eigen::Vector3f::Constant(0.5f)) * scalbnf(1.0f, level) + Eigen::Vector3f::Constant(0.5f);

    out[i] = pos;
    indices[i] = idx;
};

void generate_grid_samples_cuda(
        const torch::Tensor density_grid,
        const int density_grid_ema_step,
        const int n_elements,
        const int max_cascade,
        const int n_grid,
        const float thresh,
        torch::Tensor density_grid_positions_uniform,
        torch::Tensor density_grid_indices_uniform) {

    cudaStream_t stream = 0;

    // input value
    float* density_grid_p = (float*)density_grid.data_ptr();

    // output value
    uint32_t* density_grid_indices_p = (uint32_t*)density_grid_indices_uniform.data_ptr();
    Vector3f* density_grid_positions_uniform_p = (Vector3f*)density_grid_positions_uniform.data_ptr();

    linear_kernel(generate_grid_samples_cuda_kernel, 0, stream,
        n_elements, rng, (uint32_t)density_grid_ema_step, density_grid_p,
        density_grid_positions_uniform_p, density_grid_indices_p,
        (uint32_t)max_cascade+1, (uint32_t)n_grid, thresh);

    rng.advance();
    cudaDeviceSynchronize();
}


// -------------------------------------------------- ------------------------------------ //


__global__ void count_bitfield_cuda_kernel(
    const uint32_t n_elements,
    const uint8_t *__restrict__ density_grid_bitfield,
    float *__restrict__ counter)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n_elements)
        return;

    #pragma unroll
    for (uint8_t j = 0; j < 8; ++j)
    {
        if ((density_grid_bitfield[i] && (uint8_t)1 << j) > 0) {
            atomicAdd(&counter[0], 1.0f);
        }
    }

};

void count_bitfield_cuda(
        const torch::Tensor density_grid_bitfield,
        const torch::Tensor counter,
        const int grid,
        const int level) {

    cudaStream_t stream = 0;

    // input value
    uint32_t n_elements = (uint32_t)grid * (uint32_t)grid * (uint32_t)grid / 8;
    uint32_t offset = n_elements * (uint32_t)level;
    uint8_t* density_grid_bitfield_p = (uint8_t*)density_grid_bitfield.data_ptr();

    // output
    float* counter_p = (float*)counter.data_ptr();

    linear_kernel(count_bitfield_cuda_kernel, 0, stream, n_elements, density_grid_bitfield_p + offset, counter_p);

    cudaDeviceSynchronize();
}


// -------------------------------------------------- ------------------------------------ //


__global__ void get_occ_pc_cuda_kernel(
    const uint32_t n_elements,
    const uint8_t *__restrict__ density_grid_bitfield,
    float *__restrict__ pc,
    float *__restrict__ counter,
    uint32_t n_grid)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n_elements)
        return;

    #pragma unroll
    for (uint8_t j = 0; j < 8; ++j)
    {
        if ((density_grid_bitfield[i] && (uint8_t)1 << j) > 0) {
            uint32_t base = atomicAdd(counter, 1.0f);
            pc += base * 3;

            pc[0] = (float)morton3D_invert((i * 8 + j) >> 0) / (float)n_grid;
            pc[1] = (float)morton3D_invert((i * 8 + j) >> 1) / (float)n_grid;
            pc[2] = (float)morton3D_invert((i * 8 + j) >> 2) / (float)n_grid;

            // reset
            pc -= base * 3;
        }
    }

};

void get_occ_pc_cuda(
        const torch::Tensor density_grid_bitfield,
        torch::Tensor pc,
        torch::Tensor counter,
        const int n_grid) {

    cudaStream_t stream = 0;

    // input value
    uint32_t n_elements = (uint32_t)(n_grid * n_grid * n_grid / 8);  // only the inner one
    uint8_t* density_grid_bitfield_p = (uint8_t*)density_grid_bitfield.data_ptr();

    // output
    float* pc_p = (float*)pc.data_ptr();
    float* counter_p = (float*)counter.data_ptr();

    linear_kernel(get_occ_pc_cuda_kernel, 0, stream,
        n_elements, density_grid_bitfield_p, pc_p, counter_p, (uint32_t)n_grid);

    cudaDeviceSynchronize();
}