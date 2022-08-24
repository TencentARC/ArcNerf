// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// spherical harmonics embedding of direction xyz

#include <torch/extension.h>

#include "helper.h"


// forward table
template <typename T>
__device__ T forward_table(uint32_t c, T x, T y, T z) {
    T xx = square(x);
    T yy = square(y);
    T zz = square(z);
    T xy = x * y;
    T yz = y * z;
    T xz = x * z;

	switch(c) {
	    case 0: return 0.28209479177387814; break;
	    case 1: return -0.4886025119029199 * y; break;
	    case 2: return 0.4886025119029199 * z; break;
	    case 3: return -0.4886025119029199 * x; break;
	    case 4: return 1.0925484305920792 * xy; break;
	    case 5: return -1.0925484305920792 * yz; break;
	    case 6: return 0.31539156525252005 * (3.0 * zz - 1.0); break;
	    case 7: return -1.0925484305920792 * xz; break;
	    case 8: return 0.5462742152960396 * (xx - yy); break;
	    case 9: return -0.5900435899266435 * y * (3.0 * xx - yy); break;
	    case 10: return 2.890611442640554 * xy * z; break;
	    case 11: return -0.4570457994644658 * y * (5.0 * zz - 1.0); break;
	    case 12: return 0.3731763325901154 * z * (5.0 * zz - 3.0); break;
	    case 13: return -0.4570457994644658 * x * (5.0 * zz - 1.0); break;
	    case 14: return 1.445305721320277 * z * (xx - yy); break;
	    case 15: return -0.5900435899266435 * x * (xx - 3.0 * yy); break;
	    case 16: return 2.5033429417967046 * xy * (xx - yy); break;
	    case 17: return -1.7701307697799304 * yz * (3.0 * xx - yy); break;
	    case 18: return 0.9461746957575601 * xy * (7.0 * zz - 1.0); break;
	    case 19: return -0.6690465435572892 * yz * (7.0 * zz - 3.0); break;
	    case 20: return 0.10578554691520431 * (zz * (35.0 * zz - 30.0) + 3.0); break;
	    case 21: return -0.6690465435572892 * xz * (7.0 * zz - 3.0); break;
	    case 22: return 0.47308734787878004 * (xx - yy) * (7.0 * zz - 1.0); break;
	    case 23: return -1.7701307697799304 * xz * (xx - 3.0 * yy); break;
	    case 24: return 0.6258357354491761 * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)); break;
	    default: return 0.0;
	}
}


// The real cuda forward_kernel
template <typename scalar_t, uint32_t D_START, uint32_t D_END>
__device__ void forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    const uint32_t n) {

    # pragma unroll
    for (uint32_t idx = D_START; idx < D_END; idx++) {
        output[n][idx] = forward_table(idx, xyz[n][0], xyz[n][1], xyz[n][2]);
    }
}

// The forward wrapper
template <typename scalar_t>
__global__ void forward_kernel_wrapper(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz,
    const uint32_t degree,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output) {

    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;  // d=0,1,2,3,4,5,6
    const uint32_t n = blockIdx.y * blockDim.y + threadIdx.y;  // row id

    if (d < degree && n < xyz.size(0)) {
        switch (d) {
            case 0: forward_kernel<scalar_t, 0, 1>(xyz, output, n); break;
            case 1: forward_kernel<scalar_t, 1, 4>(xyz, output, n); break;
            case 2: forward_kernel<scalar_t, 4, 9>(xyz, output, n); break;
            case 3: forward_kernel<scalar_t, 9, 16>(xyz, output, n); break;
            case 4: forward_kernel<scalar_t, 16, 25>(xyz, output, n); break;
        }
    }
}


/* CUDA instantiate func for sh_encode forward
   @param: xyz, torch float tensor of (B, 3)
   @param: degree, int num
   @return: output, torch float tensor of (B, degree**2)
*/
torch::Tensor sh_encode_forward_cuda(
    const torch::Tensor xyz, const uint32_t degree) {

    torch::Tensor output = torch::zeros({xyz.size(0), square(degree)}, xyz.dtype()).to(xyz.device());

    const uint32_t n_row = xyz.size(0);  // B
    const uint32_t n_col = xyz.size(1);  // 3
    const uint32_t threads_per_row = 1024;
    const dim3 threads(1, threads_per_row);
    const dim3 blocks(degree, div_round_up(n_row, threads_per_row));

    // instantiate the real executable kernel
    AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "sh_encode_forward_cuda",
    ([&] {
        forward_kernel_wrapper<scalar_t><<<blocks, threads>>>(
            xyz.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            degree,
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    return output;
}


// backward table
template <typename T>
__device__ T backward_table(uint32_t c, uint32_t index, T x, T y, T z) {
    T xx = square(x);
    T yy = square(y);
    T zz = square(z);
    T xy = x * y;
    T yz = y * z;
    T xz = x * z;
    T xyz = x * y * z;

    if (index == 0) {
        switch(c) {
            case 0: return 0.0; break;
            case 1: return 0.0; break;
            case 2: return 0.0; break;
            case 3: return -0.4886025119029199; break;
            case 4: return 1.0925484305920792 * y; break;
            case 5: return 0.0; break;
            case 6: return 0.0; break;
            case 7: return -1.0925484305920792 * z; break;
            case 8: return 0.5462742152960396 * 2.0 * x; break;
            case 9: return -0.5900435899266435 * 6.0 * xy; break;
	        case 10: return 2.890611442640554 * yz; break;
	        case 11: return 0.0; break;
	        case 12: return 0.0; break;
	        case 13: return -0.4570457994644658 * (5.0 * zz - 1.0); break;
	        case 14: return 1.445305721320277 * 2.0 * xz; break;
	        case 15: return -0.5900435899266435 * 3.0 * (xx - yy); break;
	        case 16: return 2.5033429417967046 * y * (3.0* xx - yy); break;
	        case 17: return -1.7701307697799304 * 6.0 * xyz; break;
	        case 18: return 0.9461746957575601 * y * (7.0 * zz - 1.0); break;
	        case 19: return 0.0; break;
	        case 20: return 0.0; break;
	        case 21: return -0.6690465435572892 * z * (7.0 * zz - 3.0); break;
	        case 22: return 0.47308734787878004 * 2.0 * x * (7.0 * zz - 1.0); break;
	        case 23: return -1.7701307697799304 * 3.0 * z * (xx - yy); break;
	        case 24: return 0.6258357354491761 * 4.0 * x * (xx - 3.0 * yy); break;
	        default: return 0.0;
        }
    } else if (index == 1) {
        switch(c) {
            case 0: return 0.0; break;
            case 1: return -0.4886025119029199; break;
            case 2: return 0.0; break;
            case 3: return 0.0; break;
            case 4: return 1.0925484305920792 * x; break;
            case 5: return -1.0925484305920792 * z; break;
            case 6: return 0.0; break;
            case 7: return 0.0; break;
            case 8: return -0.5462742152960396 * 2.0 * y; break;
            case 9: return -0.5900435899266435 * 3.0 * (xx - yy); break;
	        case 10: return 2.890611442640554 * xz; break;
	        case 11: return -0.4570457994644658 * (5.0 * zz - 1.0); break;
            case 12: return 0.0; break;
            case 13: return 0.0; break;
	        case 14: return -1.445305721320277 * 2.0 * yz; break;
	        case 15: return 0.5900435899266435 * 6.0 * xy; break;
	        case 16: return 2.5033429417967046 * x * (xx - 3.0 * yy); break;
	        case 17: return -1.7701307697799304 * 3.0 * z * (xx - yy); break;
	        case 18: return 0.9461746957575601 * x * (7.0 * zz - 1.0); break;
	        case 19: return -0.6690465435572892 * z * (7.0 * zz - 3.0); break;
	        case 20: return 0.0; break;
	        case 21: return 0.0; break;
	        case 22: return 0.47308734787878004 * 2.0 * y * (1.0 - 7.0 * zz); break;
	        case 23: return 1.7701307697799304 * 6.0 * xyz; break;
	        case 24: return 0.6258357354491761 * 4.0 * y * (yy - 3.0 * xx); break;
	        default: return 0.0;
        }
    } else if (index == 2) {
        switch(c) {
            case 0: return 0.0; break;
            case 1: return 0.0; break;
            case 2: return 0.4886025119029199; break;
            case 3: return 0.0; break;
            case 4: return 0.0; break;
            case 5: return -1.0925484305920792 * y; break;
            case 6: return 0.31539156525252005 * 6.0 * z; break;
            case 7: return -1.0925484305920792 * x; break;
            case 8: return 0.0; break;
            case 9: return 0.0; break;
	        case 10: return 2.890611442640554 * xy; break;
	        case 11: return -0.4570457994644658 * 10.0 * yz; break;
	        case 12: return 0.3731763325901154 * (15.0 * zz - 3.0); break;
	        case 13: return -0.4570457994644658 * 10.0 * xz; break;
	        case 14: return 1.445305721320277 * (xx - yy); break;
	        case 15: return 0.0; break;
	        case 16: return 0.0; break;
	        case 17: return -1.7701307697799304 * y * (3.0 * xx - yy); break;
	        case 18: return 0.9461746957575601 * 14.0 * xyz; break;
	        case 19: return -0.6690465435572892 * 3.0 * y * (7.0 * zz - 1.0); break;
	        case 20: return 0.10578554691520431 * 20.0 * z * (7.0 * zz - 3.0); break;
	        case 21: return -0.6690465435572892 * 3.0 * x * (7.0 * zz - 1.0); break;
	        case 22: return 0.47308734787878004 * 14.0 * z * (xx - yy); break;
	        case 23: return -1.7701307697799304 * x * (xx - 3.0 * yy); break;
	        case 24: return 0.0; break;
	        default: return 0.0;
        }
    }

    return 0.0;

}


// The real cuda backward_kernel
template <typename scalar_t, uint32_t D_START, uint32_t D_END>
__device__ void backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_out,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_xyz,
    const uint32_t n) {

    scalar_t x = xyz[n][0];
    scalar_t y = xyz[n][1];
    scalar_t z = xyz[n][2];

    # pragma unroll
    for (uint32_t idx = D_START; idx < D_END; idx++) {
        // Not sure when atomicAdd harms the performance
        atomicAdd(&grad_xyz[n][0], grad_out[n][idx] * backward_table(idx, 0, x, y, z));
        atomicAdd(&grad_xyz[n][1], grad_out[n][idx] * backward_table(idx, 1, x, y, z));
        atomicAdd(&grad_xyz[n][2], grad_out[n][idx] * backward_table(idx, 2, x, y, z));
    }
}

// The backward wrapper
template <typename scalar_t>
__global__ void backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_out,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz,
    const uint32_t degree,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_xyz) {

    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;  // d=0,1,2,3,4,5,6
    const uint32_t n = blockIdx.y * blockDim.y + threadIdx.y;  // row id

    if (d < degree && n < xyz.size(0)) {
        switch (d) {
            case 0: backward_kernel<scalar_t, 0, 1>(grad_out, xyz, grad_xyz, n); break;
            case 1: backward_kernel<scalar_t, 1, 4>(grad_out, xyz, grad_xyz, n); break;
            case 2: backward_kernel<scalar_t, 4, 9>(grad_out, xyz, grad_xyz, n); break;
            case 3: backward_kernel<scalar_t, 9, 16>(grad_out, xyz, grad_xyz, n); break;
            case 4: backward_kernel<scalar_t, 16, 25>(grad_out, xyz, grad_xyz, n); break;
        }
    }
}


/* CUDA instantiate func for scale_exp backward
   @param: grad_out, torch float tensor of (B, degree**2), final grad
   @param: xyz, torch float tensor of (B, 3)
   @param: degree, int num
   @return: grad_xyz, torch float tensor of (B, 3)
*/
torch::Tensor sh_encode_backward_cuda(
    const torch::Tensor grad_out, const torch::Tensor xyz, const uint32_t degree) {

    torch::Tensor grad_xyz = torch::zeros_like(xyz).to(xyz.device());  // (B, 3)

    const uint32_t n_row = xyz.size(0);  // B
    const uint32_t n_col = xyz.size(1);  // 3
    const uint32_t threads_per_row = 512;
    const dim3 threads(1, threads_per_row);
    const dim3 blocks(degree, div_round_up(n_row, threads_per_row));

    // instantiate the real executable kernel
    AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "sh_encode_backward_cuda",
    ([&] {
        backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            xyz.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            degree,
            grad_xyz.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    return grad_xyz;
}
