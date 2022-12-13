// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// utils func


#ifndef utils_h
#define utils_h

#include <torch/extension.h>


////////////////////////////////////////////////////////////////////////////////
// Check function
////////////////////////////////////////////////////////////////////////////////
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_BYTE(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Byte, #x " must be an uint8 tensor")
#define CHECK_IS_LONG(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, #x " must be an long(int64) tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || \
                                         x.scalar_type() == at::ScalarType::Half || \
                                         x.scalar_type() == at::ScalarType::Double, \
                                         #x " must be a floating tensor")
#define CHECK_IS_BOOL(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Bool, #x " must be an bool tensor")




////////////////////////////////////////////////////////////////////////////////
// end
////////////////////////////////////////////////////////////////////////////////

#endif
