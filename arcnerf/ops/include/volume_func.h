// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// Some volume func


#include "helper_math.h"

// aabb intersection
inline __device__ float2 ray_aabb_intersect(
    const float3 rays_o,
    const float3 rays_d,
    const float3 xyz_min,
    const float3 xyz_max
){

    const float3 inv_d = 1.0f / rays_d;
    const float3 t_min = (xyz_min - rays_o) * inv_d;
    const float3 t_max = (xyz_max - rays_o) * inv_d;

    const float3 _t1 = fminf(t_min, t_max);
    const float3 _t2 = fmaxf(t_min, t_max);
    const float t1 = fmaxf(fmaxf(_t1.x, _t1.y), _t1.z);
    const float t2 = fminf(fminf(_t2.x, _t2.y), _t2.z);

    if (t1 > t2) return make_float2(-1.0f); // no intersection

    return make_float2(t1, t2);
}


// get voxel index from xyz, return -1 for invalid pts
inline __device__ float3 cal_voxel_idx_from_xyz(
    const float3 xyz,
    const float3 xyz_min,
    const float3 xyz_max,
    const float n_grid
) {

    float3 voxel_size = (xyz_max - xyz_min) / n_grid;
    float3 voxel_idx = (xyz - xyz_min) / voxel_size;
    if (voxel_idx.x < 0 || voxel_idx.y < 0 || voxel_idx.z < 0 || \
        voxel_idx.x > n_grid || voxel_idx.x > n_grid || voxel_idx.x > n_grid) {
        return make_float3(-1.0f, -1.0f, -1.0f);
    }

    return voxel_idx;
}
