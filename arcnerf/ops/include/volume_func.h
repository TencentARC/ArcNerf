// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// Some volume func


#include "helper_math.h"
#include "utils.h"


// Used to index into the PRNG stream. Must be larger than the number of
// samples consumed by any given training ray.
inline constexpr __device__ __host__ uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() { return 8; }


// aabb intersection.
inline __device__ float2 ray_aabb_intersect(
    const float3 rays_o,
    const float3 rays_d,
    const float3 xyz_min,
    const float3 xyz_max
){
    // handles invalid
    float eps = 1e-7;
    if (fabs(rays_d.x) < eps && (rays_o.x < xyz_min.x || rays_o.x > xyz_max.x))
        return make_float2(-1.0f);
    if (fabs(rays_d.y) < eps && (rays_o.y < xyz_min.y || rays_o.y > xyz_max.y))
        return make_float2(-1.0f);
    if (fabs(rays_d.z) < eps && (rays_o.z < xyz_min.z || rays_o.z > xyz_max.z))
        return make_float2(-1.0f);

    const float3 inv_d = 1.0f / rays_d;
    const float3 t_min = (xyz_min - rays_o) * inv_d;
    const float3 t_max = (xyz_max - rays_o) * inv_d;

    float t1 = fmaxf3(fminf(t_min, t_max));
    float t2 = fminf3(fmaxf(t_min, t_max));

    if (t1 > t2) return make_float2(-1.0f); // no intersection
    t1 = fmaxf(0.0f, t1);
    t2 = fmaxf(0.0f, t2);

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
    if (fminf3(voxel_idx) < 0 || fmaxf3(voxel_idx) > n_grid) {
        return make_float3(-1.0f, -1.0f, -1.0f);
    }

    return voxel_idx;
}

// check pts bounding in aabb
inline __device__ bool check_pts_in_aabb(
    const float3 xyz,
    const float3 xyz_min,
    const float3 xyz_max
) {
    if (xyz.x >= xyz_min.x && xyz.y >= xyz_min.y && xyz.z >= xyz_min.z &&\
        xyz.x <= xyz_max.x && xyz.y <= xyz_max.y && xyz.z <= xyz_max.z)
        return true;

    return false;
}

// get point from rays_o, rays_d and zvals
inline __device__ float3 get_ray_points_by_zvals(
    const float3 rays_o,
    const float3 rays_d,
    const float zval
) {
    return rays_o + zval * rays_d;
}


// update to next voxel find pos. TODO: Check correct for pts not in (0, 1)
inline __device__ float distance_to_next_voxel(
    const float3 pos,
    const float3 rays_d,
    const float3 xyz_min,
    const float3 xyz_max,
    const uint32_t n_grid
) {
    float3 xyz_center = (xyz_min + xyz_max) / 2.0;
    float3 xyz_half_len = (xyz_max - xyz_min) / 2.0;
    const float3 inv_d = 1.0f / rays_d;

	float3 p = n_grid * pos;
	float tx = (floorf(p.x + xyz_center.x + xyz_half_len.x * signf(rays_d.x)) - p.x) * inv_d.x;
	float ty = (floorf(p.y + xyz_center.y + xyz_half_len.y * signf(rays_d.y)) - p.y) * inv_d.y;
	float tz = (floorf(p.z + xyz_center.z + xyz_half_len.z * signf(rays_d.z)) - p.z) * inv_d.z;
	float t = fmin(fmin(tx, ty), tz);

	return fmaxf(t / n_grid, 0.0f);
}


// update to next voxel
inline __device__ float advance_to_next_voxel(
    float t,
    const float dt,
    const float3 pos,
    const float3 rays_d,
    const float3 xyz_min,
    const float3 xyz_max,
    const uint32_t n_grid
) {
	float t_target = t + distance_to_next_voxel(pos, rays_d, xyz_min, xyz_max, n_grid);

	do { t += dt; } while (t < t_target);
	return t;
}
