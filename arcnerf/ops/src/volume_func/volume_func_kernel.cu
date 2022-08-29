// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// volume related func in cuda


#include <torch/extension.h>

#include "helper.h"
#include "volume_func.h"
#include "pcg32.h"


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

    const float3 _xyz =  make_float3(xyz[n][0], xyz[n][1], xyz[n][2]);
    const float3 xyz_min = make_float3(range[0][0], range[1][0], range[2][0]);
    const float3 xyz_max = make_float3(range[0][1], range[1][1], range[2][1]);
    float3 voxel_idx = cal_voxel_idx_from_xyz(_xyz, xyz_min, xyz_max, (scalar_t) n_grid);

    if (voxel_idx.x < 0) return;

    const bool occ = bitfield[(uint8_t)voxel_idx.x][(uint8_t)voxel_idx.y][(uint8_t)voxel_idx.z];
    if (occ) {
        output[n] = true;
    }

    return;
}

/* CUDA instantiate func for hashgrid_encode forward
   @param: xyz, torch float tensor of (B, 3)
   @param: bitfield, (N_grid, N_grid, N_grid), bool tensor indicating each voxel's occupancy
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
        const float3 _rays_o = make_float3(rays_o[n][0], rays_o[n][1], rays_o[n][2]);
        const float3 _rays_d = make_float3(rays_d[n][0], rays_d[n][1], rays_d[n][2]);
        const float3 xyz_min = make_float3(aabb_range[v][0][0], aabb_range[v][1][0], aabb_range[v][2][0]);
        const float3 xyz_max = make_float3(aabb_range[v][0][1], aabb_range[v][1][1], aabb_range[v][2][1]);
        const float2 t1t2 = ray_aabb_intersect(_rays_o, _rays_d, xyz_min, xyz_max);
        if (t1t2.y > 0) {
            near[n][v] = t1t2.x + eps;
            far[n][v] = t1t2.y - eps;
            mask[n][v] = true;
        } else {
            near[n][v] = 0.0;
            far[n][v] = 0.0;
            mask[n][v] = false;
        }

        // get ray pts
        const float3 pts_near = _rays_o + near[n][v] * _rays_d;
        const float3 pts_far = _rays_o + far[n][v] * _rays_d;
        pts[n][v][0][0] = pts_near.x;
        pts[n][v][0][1] = pts_near.y;
        pts[n][v][0][2] = pts_near.z;
        pts[n][v][1][0] = pts_far.x;
        pts[n][v][1][1] = pts_far.y;
        pts[n][v][1][2] = pts_far.z;
    }

    return;
}

/* CUDA instantiate of aabb intersection func
   @param: rays_o, ray origin, (N_rays, 3)
   @param: rays_d, ray direction, assume normalized, (N_rays, 3)
   @param: aabb_range, bbox range of volume, (N_v, 3, 2) of xyz_min/max of each volume
   @param: eps, error threshold
   @return: near, near intersection zvals. (N_rays, N_v)
   @return: far, far intersection zvals. (N_rays, N_v)
   @return: pts, intersection pts with the volume. (N_rays, N_v, 2, 3)
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


// The real cuda kernel
template <typename scalar_t>
__global__ void sparse_volume_sampling_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rays_o,  //(N_rays, 3)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rays_d,  //(N_rays, 3)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> near,  //(N_rays, 1)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> far,  //(N_rays, 1)
    const int N_pts,
    const float dt,
    pcg32 rng,
    const float near_distance,
    const bool perturb,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> aabb_range,
    const float n_grid,
    const torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> bitfield,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> zvals,  //(N_rays, N_pts)
    torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> mask) {  //(N_rays, N_pts)

    const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;  // ray id
    if (n >= rays_o.size(0)) return;

    scalar_t startt = fmaxf(near[n][0], near_distance);
    scalar_t far_end = far[n][0];
    rng.advance(n * N_MAX_RANDOM_SAMPLES_PER_RAY());

    const float3 xyz_min = make_float3(aabb_range[0][0], aabb_range[1][0], aabb_range[2][0]);
    const float3 xyz_max = make_float3(aabb_range[0][1], aabb_range[1][1], aabb_range[2][1]);

    // perturb init zvals
    if (perturb) {
        startt += dt * rng.next_float();
    }

    const float3 _rays_o = make_float3(rays_o[n][0], rays_o[n][1], rays_o[n][2]);
    const float3 _rays_d = make_float3(rays_d[n][0], rays_d[n][1], rays_d[n][2]);
    float3 inv_d = 1.0f / _rays_d;

    uint32_t j = 0;
    scalar_t t = startt;
    float3 pos;

    while (t <= far_end && j < N_pts && \\
                check_pts_in_aabb(get_ray_points_by_zvals(_rays_o, _rays_d, t), xyz_min, xyz_max)) {
        float3 voxel_idx = cal_voxel_idx_from_xyz(_rays_o + t * _rays_d, xyz_min, xyz_max, (float) n_grid);
        if (voxel_idx.x >= 0 && bitfield[(uint8_t)voxel_idx.x][(uint8_t)voxel_idx.y][(uint8_t)voxel_idx.z]) {
            // update the pts and mask
            zvals[n][j] = t;
            mask[n][j] = true;
            ++j;
            t += dt;
        } else {
            pos = get_ray_points_by_zvals(_rays_o, _rays_d, t);
            t = advance_to_next_voxel(t, dt, CONE_ANGLE(), pos, _rays_d, xyz_min, xyz_max, n_grid);
        }
    }

    // make the remaining zvals the same as last
    float last_zval;
    if (j > 0 && j < N_pts) {
        last_zval = zvals[n][j-1];
    }
    while (j > 0 && j < N_pts) {
        zvals[n][j] = last_zval;
        ++j;
    }

    return;
}


/* CUDA instantiate of sparse_volume_sampling func
   @param: rays_o, ray origin, (N_rays, 3)
   @param: rays_d, ray direction, assume normalized, (N_rays, 3)
   @param: near, near intersection zvals. (N_rays, 1)
   @param: far, far intersection zvals. (N_rays, 1)
   @param: N_pts, max num of sampling pts on each ray.
   @param: dt, fix step length
   @param: aabb_range, bbox range of volume, (3, 2) of xyz_min/max of each volume
   @param: near_distance, near distance for sampling. By default 0.0.
   @param: perturb, whether to perturb the first zval, use in training only
   @return: zvals, (N_rays, N_pts), sampled points zvals on each rays.
   @return: mask, (N_rays, N_pts), show whether each ray has intersection with the volume, BoolTensor
*/
std::vector<torch::Tensor> sparse_volume_sampling_cuda(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor near,
    const torch::Tensor far,
    const int N_pts,
    const float dt,
    const torch::Tensor aabb_range,
    const int n_grid,
    const torch::Tensor bitfield,
    const float near_distance,
    const bool perturb) {

    const uint32_t N_rays = rays_o.size(0);  // N_rays

    const uint32_t threads = 512;
    const uint32_t blocks(div_round_up(N_rays, threads));

    // Init the output tensor
    auto dtype = rays_o.dtype();
    auto device = rays_o.device();

    // random seed to perturb the first value
    pcg32 rng = pcg32{(uint64_t)623};

    torch::Tensor zvals = torch::zeros({N_rays, N_pts}, dtype).to(device);  // (N_rays, N_pts)
    torch::Tensor mask = torch::zeros({N_rays, N_pts}, torch::kBool).to(device);  // (N_rays, N_pts)

    // instantiate the real executable kernel
    AT_DISPATCH_FLOATING_TYPES(rays_o.scalar_type(), "sparse_volume_sampling_cuda",
    ([&] {
        sparse_volume_sampling_cuda_kernel<scalar_t><<<blocks, threads>>>(
            rays_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            near.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            far.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            N_pts, dt, rng, near_distance, perturb,
            aabb_range.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            n_grid,
            bitfield.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
            zvals.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            mask.packed_accessor32<bool, 2, torch::RestrictPtrTraits>()
        );
    }));

    return {zvals, mask};
}
