// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// volume related func in cuda


#include <torch/extension.h>

#include "common.h"
#include "volume_func.h"
#include "pcg32.h"


// The real cuda kernel
__global__ void check_pts_in_occ_voxel_cuda_kernel(
    const uint32_t n_elements,
    const Vector3f *__restrict__ xyz,
    const bool *__restrict__ bitfield,
    const Vector3f *__restrict__ aabb_range,
    const uint32_t n_grid,
    bool *__restrict__ output) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;

    const Vector3f pos = xyz[i];
    const Vector3f xyz_min = aabb_range[0];
    const Vector3f xyz_max = aabb_range[1];

    bool occ = density_grid_occupied_at(pos, bitfield, xyz_min, xyz_max, n_grid);

    output[i] = occ;

    return;
}

/* CUDA instantiate func for hashgrid_encode forward
   @param: xyz, torch float tensor of (B, 3)
   @param: bitfield, (N_grid, N_grid, N_grid), bool tensor indicating each voxel's occupancy
   @param: range, torch float tensor of (2, 3), range of xyz boundary
   @param: n_grid, uint8_t resolution
   @return: output, torch bool tensor of (B,)
*/
void check_pts_in_occ_voxel_cuda(
    const torch::Tensor xyz,
    const torch::Tensor bitfield,
    const torch::Tensor aabb_range,
    const int n_grid,
    torch::Tensor output) {

    const uint32_t n_elements = xyz.size(0);  // B

    cudaStream_t stream=0;

    // input
    Vector3f* xyz_p = (Vector3f*)xyz.data_ptr();
    bool* bitfield_p = (bool*)bitfield.data_ptr();
    Vector3f* aabb_range_p = (Vector3f*)aabb_range.data_ptr();
    // output
    bool* output_p = (bool*)output.data_ptr();

    linear_kernel(check_pts_in_occ_voxel_cuda_kernel, 0, stream,
        n_elements, xyz_p, bitfield_p, aabb_range_p, (uint32_t)n_grid, output_p);

    cudaDeviceSynchronize();
}


// ------------------------------------------------------------------------------------------------ //

// The real cuda kernel
__global__ void aabb_intersection_cuda_kernel(
    const uint32_t n_elements,
    const uint32_t n_rays,
    const uint32_t n_v,
    const Vector3f *__restrict__ rays_o,  //(N_rays, 3)
    const Vector3f *__restrict__ rays_d,  //(N_rays, 3)
    const Vector3f *__restrict__ aabb_range,  //(N_v, 2, 3)
    float *__restrict__ near,  //(N_rays, N_v)
    float *__restrict__ far,  //(N_rays, N_v)
    Vector3f *__restrict__ pts,  //(N_rays, N_v, 2, 3)
    bool *__restrict__ mask) {  //(N_rays, N_v)

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;  //

    if (i >= n_elements) return;

    const uint32_t rays_idx = (i / n_v);
    const uint32_t vol_idx = (i % n_v);

    const Vector3f _rays_o = rays_o[rays_idx];
    const Vector3f _rays_d = rays_d[rays_idx];
    const Vector3f xyz_min = aabb_range[vol_idx * 2];
    const Vector3f xyz_max = aabb_range[vol_idx * 2 + 1];

    Vector2f tminmax = aabb_ray_intersect(_rays_o, _rays_d, xyz_min, xyz_max);

    // move output ptr
    near += i;
    far += i;
    pts += i * 2;
    mask += i;

    if (tminmax.x() > 0) {
        near[0] = tminmax.x();
        far[0] = tminmax.y();
        mask[0] = true;
    } else {
        near[0] = 0.0;
        far[0] = 0.0;
        mask[0] = false;
    }

    // get ray pts
    const Vector3f pts_near = _rays_o + near[0] * _rays_d;
    const Vector3f pts_far = _rays_o + far[0] * _rays_d;
    pts[0] = pts_near;
    pts[1] = pts_far;

    return;
}

/* CUDA instantiate of aabb intersection func
   @param: rays_o, ray origin, (N_rays, 3)
   @param: rays_d, ray direction, assume normalized, (N_rays, 3)
   @param: aabb_range, bbox range of volume, (N_v, 2, 3) of xyz_min/max of each volume
   @return: near, near intersection zvals. (N_rays, N_v)
   @return: far, far intersection zvals. (N_rays, N_v)
   @return: pts, intersection pts with the volume. (N_rays, N_v, 2, 3)
   @return: mask, (N_rays, N_v), show whether each ray has intersection with the volume, BoolTensor
*/
void aabb_intersection_cuda(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor aabb_range,
    torch::Tensor near,
    torch::Tensor far,
    torch::Tensor pts,
    torch::Tensor mask) {

    cudaStream_t stream=0;

    const uint32_t n_rays = rays_o.size(0);  // N_rays
    const uint32_t n_v = aabb_range.size(0); // N_v

    const uint32_t n_elements = n_rays * n_v;

    // inputs
    Vector3f* rays_o_p = (Vector3f*)rays_o.data_ptr();
    Vector3f* rays_d_p = (Vector3f*)rays_d.data_ptr();
    Vector3f* aabb_range_p = (Vector3f*)aabb_range.data_ptr();

    // outputs
    float* near_p = (float*)near.data_ptr();
    float* far_p = (float*)far.data_ptr();
    Vector3f* pts_p = (Vector3f*)pts.data_ptr();
    bool* mask_p = (bool*)mask.data_ptr();

    linear_kernel(aabb_intersection_cuda_kernel, 0, stream, n_elements,
        n_rays, n_v, rays_o_p, rays_d_p, aabb_range_p,
        near_p, far_p, pts_p, mask_p);

    cudaDeviceSynchronize();

}


// ------------------------------------------------------------------------------------------------ //


// The real cuda kernel
__global__ void sparse_volume_sampling_cuda_kernel(
    const uint32_t n_rays,
    const Vector3f *__restrict__ rays_o,  //(N_rays, 3)
    const Vector3f *__restrict__ rays_d,  //(N_rays, 3)
    const float *__restrict__ near,  //(N_rays, 1)
    const float *__restrict__ far,  //(N_rays, 1)
    const Vector3f *__restrict__ aabb_range,  // (2, 3)
    const bool *__restrict__ bitfield,  //(n_grid, n_grid, n_grid)
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
        if (density_grid_occupied_at(pos, bitfield, xyz_min, xyz_max, n_grid)) {
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
   @param: near_distance, near distance for sampling. By default 0.0.
   @return: zvals, (N_rays, N_pts), sampled points zvals on each rays.
   @return: mask, (N_rays, N_pts), show whether each ray has intersection with the volume, BoolTensor
*/
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
    torch::Tensor mask) {

    cudaStream_t stream=0;

    const uint32_t n_rays = rays_o.size(0);  // N_rays

    // inputs
    Vector3f* rays_o_p = (Vector3f*)rays_o.data_ptr();
    Vector3f* rays_d_p = (Vector3f*)rays_d.data_ptr();
    Vector3f* aabb_range_p = (Vector3f*)aabb_range.data_ptr();
    float* near_p = (float*)near.data_ptr();
    float* far_p = (float*)far.data_ptr();
    bool* bitfield_p = (bool*)bitfield.data_ptr();

    // outputs
    float* zvals_p = (float*)zvals.data_ptr();
    bool* mask_p = (bool*)mask.data_ptr();

    linear_kernel(sparse_volume_sampling_cuda_kernel, 0, stream, n_rays,
        rays_o_p, rays_d_p, near_p, far_p, aabb_range_p, bitfield_p,
        (uint32_t)n_grid, (uint32_t)n_pts, dt, near_distance,
        rng, zvals_p, mask_p);

    rng.advance();
    cudaDeviceSynchronize();

}


// ------------------------------------------------------------------------------------------------ //

// The real cuda kernel
__global__ void tensor_reduce_max_cuda_kernel(
    const uint32_t n_elements,
    const float *__restrict__ full_tensor,
    const uint64_t *__restrict__ group_idx,
    const uint32_t n_group,
    float *__restrict__ uni_tensor) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;  // ray id
    if (i >= n_elements) return;

    uint64_t real_idx = group_idx[i];
    atomicMax((uint32_t *)&uni_tensor[real_idx], __float_as_uint(full_tensor[i]));
}

/* CUDA instantiate of tensor_reduce_max func
   @param: full_tensor: full value tensor, (N, )
   @param: group_idx: index of each row (N, )
   @param: n_group: num of group (N_uni)
   @return: uni_tensor: (N_uni. ) maximum of each unique group
*/
void tensor_reduce_max_cuda(
    const torch::Tensor full_tensor,
    const torch::Tensor group_idx,
    const int n_group,
    torch::Tensor uni_tensor) {

    cudaStream_t stream=0;

    const uint32_t n_elements = full_tensor.size(0);  // N

    // inputs
    float* full_tensor_p = (float*)full_tensor.data_ptr();
    uint64_t* group_idx_p = (uint64_t*)group_idx.data_ptr();

    // outputs
    float* uni_tensor_p = (float*)uni_tensor.data_ptr();

    linear_kernel(tensor_reduce_max_cuda_kernel, 0, stream, n_elements,
        full_tensor_p, group_idx_p, (uint32_t)n_group, uni_tensor_p);

}
