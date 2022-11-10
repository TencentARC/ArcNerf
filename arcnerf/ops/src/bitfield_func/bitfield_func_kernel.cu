// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// bitfield related func in cuda


#include <torch/extension.h>

#include "common.h"
#include "volume_func.h"
#include "pcg32.h"



// ------------------------------------------------------------------------------------------------ //


// The real cuda kernel
__global__ void sparse_volume_sampling_bit_cuda_kernel(
    const uint32_t n_rays,
    const Vector3f *__restrict__ rays_o,  //(N_rays, 3)
    const Vector3f *__restrict__ rays_d,  //(N_rays, 3)
    const float *__restrict__ near,  //(N_rays, 1)
    const float *__restrict__ far,  //(N_rays, 1)
    const Vector3f *__restrict__ aabb_range,  // (2, 3)
    const uint8_t *__restrict__ bitfield,  //(n_grid**3)
    const uint32_t n_grid,
    const uint32_t n_pts,
    const float dt,
    const float near_distance,
    default_rng_t rng,
    float *__restrict__ zvals,  //(N_rays, N_pts)
    bool *__restrict__ mask) {  //(N_rays, N_pts)

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;  // ray id
    if (i >= n_rays) return;

    rng.advance(i * 8);

    Vector3f _rays_o = rays_o[i];
    Vector3f _rays_d = rays_d[i];
    float startt = fmaxf(near[i], near_distance);
    float far_end = far[i];
    // perturb init zvals
    startt += dt * random_val(rng);

    const Vector3f xyz_min = aabb_range[0];
    const Vector3f xyz_max = aabb_range[1];

    uint32_t j = 0;
    float t = startt;
    Vector3f pos;

    // move zvals and mask
    zvals += i * n_pts;
    mask += i * n_pts;

    while (t <= far_end && j < n_pts && check_pts_in_aabb(pos = _rays_o + _rays_d * t, xyz_min, xyz_max)) {
        if (density_grid_occupied_at_bit(pos, bitfield, xyz_min, xyz_max, n_grid)) {
            // update the pts and mask
            zvals[j] = t;
            mask[j] = true;
            ++j;
            t += dt;
        } else {
            t = advance_to_next_voxel(t, dt, pos, _rays_d, xyz_min, xyz_max, n_grid);
        }
    }

    // make the remaining zvals the same as last
    float last_zval;
    if (j > 0 && j < n_pts) {
        last_zval = zvals[j-1];
    }
    while (j > 0 && j < n_pts) {
        zvals[j] = last_zval;
        ++j;
    }

    return;
}


/* CUDA instantiate of sparse_volume_sampling func
   @param: rays_o, ray origin, (N_rays, 3)
   @param: rays_d, ray direction, assume normalized, (N_rays, 3)
   @param: near, near intersection zvals. (N_rays, 1)
   @param: far, far intersection zvals. (N_rays, 1)
   @param: n_pts, max num of sampling pts on each ray.
   @param: dt, fix step length
   @param: aabb_range, bbox range of volume, (2, 3) of xyz_min/max of each volume
   @param: n_grid, num of grid
   @param: bitfield, (n_grid**3 / 8) uint8 bit
   @param: near_distance, near distance for sampling. By default 0.0.
   @return: zvals, (N_rays, N_pts), sampled points zvals on each rays.
   @return: mask, (N_rays, N_pts), show whether each ray has intersection with the volume, BoolTensor
*/
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
    torch::Tensor mask) {

    cudaStream_t stream=0;

    const uint32_t n_rays = rays_o.size(0);  // N_rays

    // inputs
    Vector3f* rays_o_p = (Vector3f*)rays_o.data_ptr();
    Vector3f* rays_d_p = (Vector3f*)rays_d.data_ptr();
    Vector3f* aabb_range_p = (Vector3f*)aabb_range.data_ptr();
    float* near_p = (float*)near.data_ptr();
    float* far_p = (float*)far.data_ptr();
    uint8_t* bitfield_p = (uint8_t*)bitfield.data_ptr();

    // outputs
    float* zvals_p = (float*)zvals.data_ptr();
    bool* mask_p = (bool*)mask.data_ptr();

    linear_kernel(sparse_volume_sampling_bit_cuda_kernel, 0, stream, n_rays,
        rays_o_p, rays_d_p, near_p, far_p, aabb_range_p, bitfield_p,
        (uint32_t)n_grid, (uint32_t)n_pts, dt, near_distance,
        rng, zvals_p, mask_p);

    rng.advance();
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
    uint32_t n_grid,
    const float thresh)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n_elements)
        return;

    // 1 random number to select the level, 3 to select the position.
    rng.advance(i * 4);
    uint32_t n_grid_per_level = n_grid * n_grid * n_grid;

    // Select grid cell that has density
    uint32_t idx;
    // uint32_t step=*step_p; # use input param
    for (uint32_t j = 0; j < 10; ++j)
    {
        idx = ((i + step * n_elements) * 56924617 + j * 19349663 + 96925573) % n_grid_per_level;
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

    // add noise inside voxel. It is in [0, 1) only.
    Eigen::Vector3f pos = (Eigen::Vector3f{(float)x, (float)y, (float)z} + random_val_3d(rng)) / n_grid;

    out[i] = pos;
    indices[i] = idx;
};

void generate_grid_samples_cuda(
        const torch::Tensor density_grid,
        const int density_grid_ema_step,
        const int n_elements,
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
        density_grid_positions_uniform_p, density_grid_indices_p, (uint32_t)n_grid, thresh);

    rng.advance();
    cudaDeviceSynchronize();
}


// -------------------------------------------------- ------------------------------------ //


template <typename T>
__global__ void splat_grid_samples_cuda_kernel(
    const uint32_t n_elements,
    const uint32_t *__restrict__ indices,
    const T *density_output,
    float *__restrict__ grid_out)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    uint32_t local_idx = indices[i];
    float optical_thickness = density_output[i];  // already applied activation in model

    // Positive floats are monotonically ordered when their bit pattern is interpretes as uint.
    // uint atomicMax is thus perfectly acceptable.
    atomicMax((uint32_t *)&grid_out[local_idx], __float_as_uint(optical_thickness));
}

void splat_grid_samples_cuda(
    const torch::Tensor density,
    const torch::Tensor density_grid_indices,
    const int n_density_grid_samples,
    torch::Tensor density_grid_tmp) {

    cudaStream_t stream=0;
    // input
    uint32_t u_n_density_grid_samples = (uint32_t)n_density_grid_samples;
    uint32_t* density_grid_indices_p = (uint32_t*)density_grid_indices.data_ptr();
    float* density_p = (float*)density.data_ptr();
    // output
    float* density_grid_tmp_p = (float*)density_grid_tmp.data_ptr();

    linear_kernel(splat_grid_samples_cuda_kernel<float>, 0, stream,
        u_n_density_grid_samples, density_grid_indices_p, density_p, density_grid_tmp_p);

    cudaDeviceSynchronize();
}


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

void update_bitfield_cuda(
    const torch::Tensor density_grid,
    const float density_grid_mean,
    torch::Tensor density_grid_bitfield,
    const float thres,
    const int n_grid) {

    cudaStream_t stream=0;
    // input
    float* density_grid_p = (float*)density_grid.data_ptr();
    // output
    uint8_t* density_grid_bitfield_p = (uint8_t*)density_grid_bitfield.data_ptr();

    const uint32_t u_n_grid = (uint32_t)n_grid;
    const uint32_t n_elements = u_n_grid * u_n_grid * u_n_grid;

    linear_kernel(grid_to_bitfield, 0, stream, n_elements / 8,
        density_grid_p, density_grid_bitfield_p, density_grid_mean, thres);

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
        const int grid) {

    cudaStream_t stream = 0;

    // input value
    uint32_t n_elements = (uint32_t)grid * (uint32_t)grid * (uint32_t)grid / 8;
    uint8_t* density_grid_bitfield_p = (uint8_t*)density_grid_bitfield.data_ptr();

    // output
    float* counter_p = (float*)counter.data_ptr();

    linear_kernel(count_bitfield_cuda_kernel, 0, stream, n_elements, density_grid_bitfield_p, counter_p);

    cudaDeviceSynchronize();
}
