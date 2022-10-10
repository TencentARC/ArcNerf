// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// common func


#ifndef common_h
#define common_h

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

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

inline HOST_DEVICE float clamp(float val, float lower, float upper)
{
    return val < lower ? lower : (upper < val ? upper : val);
}

inline HOST_DEVICE Vector3f floorv3f(Eigen::Vector3f xyz) {
    Eigen::Vector3f out(floorf(xyz.x()), floorf(xyz.y()), floorf(xyz.z()));
    return out;
}

inline HOST_DEVICE Vector3f signv3f(Eigen::Vector3f xyz) {
    Eigen::Vector3f out(sign(xyz.x()), sign(xyz.y()), sign(xyz.z()));
    return out;
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
