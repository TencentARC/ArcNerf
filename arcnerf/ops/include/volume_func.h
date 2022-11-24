// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// Some volume func

#include "common.h"

#define HOST_DEVICE __host__ __device__


////////////////////////////////////////////////////////////////////////////////
// volume function
////////////////////////////////////////////////////////////////////////////////

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
inline HOST_DEVICE uint32_t convert_xyz_index_to_flatten_index(Vector3f xyz_index, uint32_t n_grid) {
    uint32_t x = (uint32_t) floorf(xyz_index.x());
    uint32_t y = (uint32_t) floorf(xyz_index.y());
    uint32_t z = (uint32_t) floorf(xyz_index.z());

    uint32_t flatten_index = x * (n_grid * n_grid) + y * n_grid + z;

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
    if (voxel_idx.minCoeff() < 0 || voxel_idx.maxCoeff() >= (float)n_grid) {
        return false;
    }

    uint32_t flatten_index = convert_xyz_index_to_flatten_index(voxel_idx, n_grid);
    if (bitfield[flatten_index]) { return true; }

    return false;
}


// check pts bounding in aabb
inline HOST_DEVICE bool check_pts_in_aabb(const Vector3f xyz, const Vector3f xyz_min, const Vector3f xyz_max
) {
    return xyz.x() >= xyz_min.x() && xyz.y() >= xyz_min.y() && xyz.z() >= xyz_min.z() && \
           xyz.x() <= xyz_max.x() && xyz.y() <= xyz_max.y() && xyz.z() <= xyz_max.z();
}


// update to next voxel find pos.
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


////////////////////////////////////////////////////////////////////////////////
// volume function with bitfield.
////////////////////////////////////////////////////////////////////////////////

// morton3D coord transfer
inline HOST_DEVICE uint32_t expand_bits(uint32_t v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

inline HOST_DEVICE uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z)
{
	uint32_t xx = expand_bits(x);
	uint32_t yy = expand_bits(y);
	uint32_t zz = expand_bits(z);
	return xx | (yy << 1) | (zz << 2);
}

inline HOST_DEVICE uint32_t morton3D_invert(uint32_t x)
{
	x = x & 0x49249249;
	x = (x | (x >> 2)) & 0xc30c30c3;
	x = (x | (x >> 4)) & 0x0f00f00f;
	x = (x | (x >> 8)) & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

inline HOST_DEVICE uint32_t cascaded_grid_idx_at_bit(Vector3f pos, const Vector3f xyz_min, const Vector3f xyz_max, const uint32_t n_grid)
{
    Vector3f voxel_size = (xyz_max - xyz_min) / (float)n_grid;
    Vector3f voxel_idx = (pos - xyz_min).array() / voxel_size.array();
    Vector3i i = voxel_idx.cast<int>();

    uint32_t idx = morton3D(
        clamp(i.x(), 0, (int)n_grid - 1),
        clamp(i.y(), 0, (int)n_grid - 1),
        clamp(i.z(), 0, (int)n_grid - 1));

    return idx;
}

// check whether the pts is occupied
inline HOST_DEVICE bool density_grid_occupied_at_bit(
    const Vector3f pos,
    const uint8_t *bitfield,
    const Vector3f xyz_min,
    const Vector3f xyz_max,
    const uint32_t n_grid
) {
    uint32_t idx = cascaded_grid_idx_at_bit(pos, xyz_min, xyz_max, n_grid);

    return bitfield[idx / 8] & (1 << (idx % 8));
}

////////////////////////////////////////////////////////////////////////////////
// specially for multi-res vol excluding the inner vol
////////////////////////////////////////////////////////////////////////////////

// mip level in multi-res vol
inline __device__ int mip_from_pos(
    const Vector3f &pos,
    const Vector3f xyz_min,  // should be the inner one
    const Vector3f xyz_max,
    const uint32_t n_cascades
) {
    int exponent;
    int exponent_x;
    int exponent_y;
    int exponent_z;
    // The volume sides are not equal, need to check
    Vector3f xyz_center = (xyz_min + xyz_max) / 2.0f;
    Vector3f xyz_half_len = (xyz_max - xyz_min) / 2.0f;
    Vector3f absval_xyz = (pos - xyz_center).cwiseAbs().cwiseProduct(xyz_half_len.cwiseInverse());  // Norm by side

    frexpf(absval_xyz.x(), &exponent_x);
    frexpf(absval_xyz.y(), &exponent_y);
    frexpf(absval_xyz.z(), &exponent_z);
    // Get the max one
    exponent = max(max(exponent_x, exponent_y), exponent_z);

    return min(n_cascades - 1, max(0, exponent));
}


inline HOST_DEVICE uint32_t cascaded_grid_idx_at_multivol(
   Vector3f pos,
   const uint32_t mip,
   const Vector3f xyz_min,  // should be the inner one
   const Vector3f xyz_max,
   const uint32_t n_grid
) {
    // Norm into the inner volume
    Vector3f xyz_center = (xyz_min + xyz_max) / 2.0;
    float mip_scale = scalbnf(1.0f, -mip);
    pos -= xyz_center;
    pos *= mip_scale;
    pos += xyz_center;

    Vector3f voxel_size = (xyz_max - xyz_min) / (float)n_grid;
    Vector3f voxel_idx = (pos - xyz_min).array() / voxel_size.array();
    Vector3i i = voxel_idx.cast<int>();

    uint32_t idx = morton3D(
        clamp(i.x(), 0, (int)n_grid - 1),
        clamp(i.y(), 0, (int)n_grid - 1),
        clamp(i.z(), 0, (int)n_grid - 1));

    return idx;
}


inline HOST_DEVICE uint32_t grid_mip_offset(uint32_t mip, const uint32_t n_grid) {
	return (n_grid * n_grid * n_grid) * mip;
}


inline HOST_DEVICE bool density_grid_occupied_at_multivol(
    Vector3f &pos,
    const uint8_t *bitfield,
    const uint32_t mip,
    const Vector3f xyz_min,  // should be the inner one
    const Vector3f xyz_max,
    const uint32_t n_grid,
    const bool inclusive
) {
    uint32_t idx = cascaded_grid_idx_at_multivol(pos, mip, xyz_min, xyz_max, n_grid);

    if (inclusive)
        return bitfield[idx / 8 + grid_mip_offset(mip, n_grid) / 8] & (1 << (idx % 8));
    else
        return bitfield[idx / 8 + grid_mip_offset(mip-1, n_grid) / 8] & (1 << (idx % 8));  // ignore the first level
}


inline HOST_DEVICE float calc_dt(float t, float cone_angle, float min_step, float max_step) {
    return clamp(t * cone_angle, min_step, max_step);
}


// update to next voxel
inline HOST_DEVICE float advance_to_next_voxel_multivol(
    float t,
    const float cone_angle,
    const float min_step,
    const float max_step,
    const Vector3f pos,
    const Vector3f rays_d,
    const Vector3f xyz_min,
    const Vector3f xyz_max,
    const uint32_t n_grid
) {
    // Is the distance function works for different res?
	float t_target = t + distance_to_next_voxel(pos, rays_d, xyz_min, xyz_max, n_grid);

	do { t += calc_dt(t, cone_angle, min_step, max_step); } while (t < t_target);
	return t;
}
