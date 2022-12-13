// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// common func


#ifndef common_h
#define common_h

#include <atomic>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

#include <Eigen/Core>
#include <Eigen/Dense>
using namespace Eigen;

#define HOST_DEVICE __host__ __device__

#include "pcg32.h"
using default_rng_t = pcg32;
static pcg32 rng{9121};


////////////////////////////////////////////////////////////////////////////////
// kernel helper
////////////////////////////////////////////////////////////////////////////////
constexpr uint32_t n_threads_linear = 128;
template <typename T>
HOST_DEVICE T div_round_up(T val, T divisor)
{
	return (val + divisor - 1) / divisor;
}
template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements)
{
	return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
}
template <typename K, typename T, typename... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types... args)
{
	if (n_elements <= 0)
	{
		return;
	}
	kernel<<<n_blocks_linear(n_elements), n_threads_linear, shmem_size, stream>>>((uint32_t)n_elements, args...);
}


////////////////////////////////////////////////////////////////////////////////
// helper func
////////////////////////////////////////////////////////////////////////////////
template <typename T>
HOST_DEVICE void host_device_swap(T &a, T &b)
{
    T c(a);
    a = b;
    b = c;
}

inline HOST_DEVICE float sign(float x)
{
    return copysignf(1.0, x);
}

HOST_DEVICE inline float clamp(float val, float lower, float upper)
{
    return val < lower ? lower : (upper < val ? upper : val);
}


////////////////////////////////////////////////////////////////////////////////
// geometry
////////////////////////////////////////////////////////////////////////////////

HOST_DEVICE Vector2f aabb_ray_intersect(const Vector2f aabb_range, const Vector3f &pos, const Vector3f &dir)
{
    float tmin = (aabb_range.x() - pos.x()) / dir.x();
    float tmax = (aabb_range.y() - pos.x()) / dir.x();

    if (tmin > tmax)
    {
        host_device_swap(tmin, tmax);
    }

    float tymin = (aabb_range.x() - pos.y()) / dir.y();
    float tymax = (aabb_range.y() - pos.y()) / dir.y();

    if (tymin > tymax)
    {
        host_device_swap(tymin, tymax);
    }

    if (tmin > tymax || tymin > tmax)
    {
        return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    }

    if (tymin > tmin)
    {
        tmin = tymin;
    }

    if (tymax < tmax)
    {
        tmax = tymax;
    }

    float tzmin = (aabb_range.x() - pos.z()) / dir.z();
    float tzmax = (aabb_range.y() - pos.z()) / dir.z();

    if (tzmin > tzmax)
    {
        host_device_swap(tzmin, tzmax);
    }

    if (tmin > tzmax || tzmin > tmax)
    {
        return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    }

    if (tzmin > tmin)
    {
        tmin = tzmin;
    }

    if (tzmax < tmax)
    {
        tmax = tzmax;
    }

    return {tmin, tmax};
}

HOST_DEVICE bool bbox_contains(const Vector2f aabb_range, const Vector3f &p)
{
    return p.x() >= aabb_range.x() && p.x() <= aabb_range.y() &&
           p.y() >= aabb_range.x() && p.y() <= aabb_range.y() &&
           p.z() >= aabb_range.x() && p.z() <= aabb_range.y();
}


HOST_DEVICE inline uint32_t expand_bits(uint32_t v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

HOST_DEVICE inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z)
{
	uint32_t xx = expand_bits(x);
	uint32_t yy = expand_bits(y);
	uint32_t zz = expand_bits(z);
	return xx | (yy << 1) | (zz << 2);
}

HOST_DEVICE inline uint32_t morton3D_invert(uint32_t x)
{
	x = x & 0x49249249;
	x = (x | (x >> 2)) & 0xc30c30c3;
	x = (x | (x >> 4)) & 0x0f00f00f;
	x = (x | (x >> 8)) & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

inline HOST_DEVICE float calc_dt(float t, float cone_angle, float min_step, float max_step) {
    return clamp(t * cone_angle, min_step, max_step);
}

inline HOST_DEVICE uint32_t grid_mip_offset(uint32_t mip, const uint32_t n_grid) {
	return (n_grid * n_grid * n_grid) * mip;
}

inline __device__ int mip_from_pos(const Vector3f &pos, const uint32_t n_cascades)
{
    int exponent;
    float maxval = (pos - Vector3f::Constant(0.5f)).cwiseAbs().maxCoeff();
    frexpf(maxval, &exponent);
    return min(n_cascades - 1, max(0, exponent + 1));
}

inline __device__ int mip_from_dt(float dt, const Vector3f &pos, const uint32_t n_grid, const uint32_t n_cascades)
{
    int mip = mip_from_pos(pos, n_cascades);
    dt *= 2 * n_grid;
    if (dt < 1.f)
        return mip;
    int exponent;
    frexpf(dt, &exponent);
    return min(n_cascades - 1, max(exponent, mip));
}

inline __device__ float distance_to_next_voxel(const Vector3f &pos, const Vector3f &dir, const Vector3f &idir, uint32_t res)
{ // dda like step
    Vector3f p = res * pos;
    float tx = (floorf(p.x() + 0.5f + 0.5f * sign(dir.x())) - p.x()) * idir.x();
    float ty = (floorf(p.y() + 0.5f + 0.5f * sign(dir.y())) - p.y()) * idir.y();
    float tz = (floorf(p.z() + 0.5f + 0.5f * sign(dir.z())) - p.z()) * idir.z();
    float t = min(min(tx, ty), tz);

    return fmaxf(t / res, 0.0f);
}

inline __device__ float advance_to_next_voxel(float t, float cone_angle, float min_step, float max_step, const Vector3f &pos, const Vector3f &dir, const Vector3f &idir, uint32_t res)
{
    // Regular stepping (may be slower but matches non-empty space)
    float t_target = t + distance_to_next_voxel(pos, dir, idir, res);
    do
    {
        t += calc_dt(t, cone_angle, min_step, max_step);
    } while (t < t_target);
    return t;
}

inline __device__ uint32_t cascaded_grid_idx_at(Vector3f pos, uint32_t mip, const uint32_t n_grid)
{
    float mip_scale = scalbnf(1.0f, -mip);
    pos -= Vector3f::Constant(0.5f);
    pos *= mip_scale;
    pos += Vector3f::Constant(0.5f);

    Vector3i i = (pos * n_grid).cast<int>();

    uint32_t idx = morton3D(
        clamp(i.x(), 0, (int)n_grid - 1),
        clamp(i.y(), 0, (int)n_grid - 1),
        clamp(i.z(), 0, (int)n_grid - 1));

    return idx;
}

inline __device__ bool density_grid_occupied_at(
    const Vector3f &pos, const uint8_t *density_grid_bitfield, uint32_t mip, const uint32_t n_grid)
{
    uint32_t idx = cascaded_grid_idx_at(pos, mip, n_grid);
    return density_grid_bitfield[idx / 8 + grid_mip_offset(mip, n_grid) / 8] & (1 << (idx % 8));
}

////////////////////////////////////////////////////////////////////////////////
// random val
////////////////////////////////////////////////////////////////////////////////

template <typename RNG>
inline HOST_DEVICE float random_val(RNG &rng)
{
	return rng.next_float();
}

template <typename RNG>
inline HOST_DEVICE Eigen::Vector3f random_val_3d(RNG &rng)
{
	return {rng.next_float(), rng.next_float(), rng.next_float()};
}

__device__ inline float random_val(uint32_t seed, uint32_t idx)
{
    pcg32 rng(((uint64_t)seed << 32) | (uint64_t)idx);
    return rng.next_float();
}

template <typename RNG>
inline __host__ __device__ Eigen::Vector2f random_val_2d(RNG &rng)
{
    return {rng.next_float(), rng.next_float()};
}

////////////////////////////////////////////////////////////////////////////////
// end
////////////////////////////////////////////////////////////////////////////////

#endif