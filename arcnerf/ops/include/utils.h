// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// utils func

#include <cuda_runtime.h>

#include <torch/extension.h>

// Check func on device
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_LONG(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, #x " must be an long(int64) tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || \
                                         x.scalar_type() == at::ScalarType::Half || \
                                         x.scalar_type() == at::ScalarType::Double, \
                                         #x " must be a floating tensor")
#define CHECK_IS_BOOL(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Bool, #x " must be an bool tensor")


// Any std::vector to gpu memory ptr
template <typename T>
T* vec_to_gpu(const std::vector<T> vec_cpu) {
    T* vec_gpu;
    auto memory_size = sizeof(T) * vec_cpu.size();
    cudaMalloc((void**)& vec_gpu, memory_size);
    cudaMemcpy(vec_gpu, &vec_cpu[0], memory_size, cudaMemcpyHostToDevice);

    return vec_gpu;
}
