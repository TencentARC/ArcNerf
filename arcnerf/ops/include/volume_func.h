// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// Some volume func

#include "common.h"

#define HOST_DEVICE __host__ __device__


// aabb intersection.
HOST_DEVICE Vector2f aabb_ray_intersect(
    const Vector3f pos,
    const Vector3f dir,
    const Vector3f xyz_min,
    const Vector3f xyz_max)
{
    // x dim
    float tmin = (xyz_min.x() - pos.x()) / dir.x();
    float tmax = (xyz_max.x() - pos.x()) / dir.x();

    if (tmin > tmax) { host_device_swap(tmin, tmax); }

    // y dim
    float tymin = (xyz_min.y() - pos.y()) / dir.y();
    float tymax = (xyz_max.y() - pos.y()) / dir.y();
    if (tymin > tymax) { host_device_swap(tymin, tymax); }

    if (tmin > tymax || tymin > tmax)
    {
        return {-1.0f, -1.0f};
    }

    if (tymin > tmin) { tmin = tymin; }
    if (tymax < tmax) { tmax = tymax; }

    // z dim
    float tzmin = (xyz_min.z() - pos.z()) / dir.z();
    float tzmax = (xyz_max.z() - pos.z()) / dir.z();
    if (tzmin > tzmax) { host_device_swap(tzmin, tzmax); }

    if (tmin > tzmax || tzmin > tmax)
    {
        return {-1.0f, -1.0f};
    }

    if (tzmin > tmin) { tmin = tzmin; }
    if (tzmax < tmax) { tmax = tzmax; }

    return {tmin, tmax};
}

// convert 3d xyz index to 1d index. We don't use the morton code, which is not consistent with torch volume
inline HOST_DEVICE float convert_xyz_index_to_flatten_index(Vector3f xyz_index, uint32_t n_grid) {
    float x = xyz_index.x();
    float y = xyz_index.y();
    float z = xyz_index.z();

    float n = (float)n_grid;
    float flatten_index = x * (n * n) + y * n + z;

    return flatten_index;
}


// check whether the pts is occupied
inline HOST_DEVICE bool density_grid_occupied_at(
    const Vector3f xyz,
    const bool *bitfield,
    const Vector3f xyz_min,
    const Vector3f xyz_max,
    const uint32_t n_grid
) {
    Vector3f voxel_size = (xyz_max - xyz_min) / (float)n_grid;
    Vector3f voxel_idx = (xyz - xyz_min).array() / voxel_size.array();
    if (voxel_idx.minCoeff() < 0 || voxel_idx.maxCoeff() > (float)n_grid) {
        return false;
    }

    uint32_t flatten_index = (uint32_t)convert_xyz_index_to_flatten_index(voxel_idx, n_grid);
    if (bitfield[flatten_index]) { return true; }

    return false;
}


// check pts bounding in aabb
inline HOST_DEVICE bool check_pts_in_aabb(const Vector3f xyz, const Vector3f xyz_min, const Vector3f xyz_max
) {
    return xyz.x() >= xyz_min.x() && xyz.y() >= xyz_min.y() && xyz.z() >= xyz_min.z() && \
           xyz.x() <= xyz_max.x() && xyz.y() <= xyz_max.y() && xyz.z() <= xyz_max.z();
}


// update to next voxel find pos. TODO: Check correct for pts not in (0, 1)
inline HOST_DEVICE float distance_to_next_voxel(
    const Vector3f pos,
    const Vector3f rays_d,
    const Vector3f xyz_min,
    const Vector3f xyz_max,
    const uint32_t n_grid
) {
    Vector3f xyz_center = (xyz_min + xyz_max) / 2.0;
    Vector3f xyz_half_len = (xyz_max - xyz_min) / 2.0;
    const Vector3f inv_d = rays_d.cwiseInverse();

	Vector3f p = (float)n_grid * pos;
	Vector3f sign_d = signv3f(rays_d);
    Vector3f t = (floorv3f(p + xyz_center + xyz_half_len.cwiseProduct(signv3f(rays_d))) - p).cwiseProduct(inv_d);
	float t_min = t.minCoeff();

	return fmaxf(t_min / n_grid, 0.0f);
}


// update to next voxel
inline HOST_DEVICE float advance_to_next_voxel(
    float t,
    const float dt,
    const Vector3f pos,
    const Vector3f rays_d,
    const Vector3f xyz_min,
    const Vector3f xyz_max,
    const uint32_t n_grid
) {
	float t_target = t + distance_to_next_voxel(pos, rays_d, xyz_min, xyz_max, n_grid);

	do { t += dt; } while (t < t_target);
	return t;
}
