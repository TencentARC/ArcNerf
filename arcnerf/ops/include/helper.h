// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// helper func


// CUDA function for simple calculation on any type
template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}


// square func
template <typename T>
inline __host__ __device__ T square(T val) {
	return val * val;
}
