#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>


// CUDA function for simple calculation on any type
template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename T>
__host__ __device__ T square(T val) {
	return val * val;
}

// forward table
template <typename T>
__device__ T forward_table(uint32_t c, T x, T y, T z) {
    T xx = square(x);
    T yy = square(y);
    T zz = square(z);
    T xy = x * y;
    T yz = y * z;
    T xz = x * z;

	switch(c){
	    case 0: return 0.28209479177387814;
	    case 1: return -0.4886025119029199 * y;
	    case 2: return 0.4886025119029199 * z;
	    case 3: return -0.4886025119029199 * x;
	    case 4: return 1.0925484305920792 * xy;
	    case 5: return -1.0925484305920792 * yz;
	    case 6: return 0.31539156525252005 * (3.0 * zz - 1.0);
	    case 7: return -1.0925484305920792 * xz;
	    case 8: return 0.5462742152960396 * (xx - yy);
	    case 9: return -0.5900435899266435 * y * (3.0 * xx - yy);
	    case 10: return 2.890611442640554 * xy * z;
	    case 11: return -0.4570457994644658 * y * (5.0 * zz - 1.0);
	    case 12: return 0.3731763325901154 * z * (5.0 * zz - 3.0);
	    case 13: return -0.4570457994644658 * x * (5.0 * zz - 1.0);
	    case 14: return 1.445305721320277 * z * (xx - yy);
	    case 15: return -0.5900435899266435 * x * (xx - 3.0 * yy);
	    case 16: return 2.5033429417967046 * xy * (xx - yy);
	    case 17: return -1.7701307697799304 * yz * (3.0 * xx - yy);
	    case 18: return 0.9461746957575601 * xy * (7.0 * zz - 1.0);
	    case 19: return -0.6690465435572892 * yz * (7.0 * zz - 3.0);
	    case 20: return 0.10578554691520431 * (zz * (35.0 * zz - 30.0) + 3.0);
	    case 21: return -0.6690465435572892 * xz * (7.0 * zz - 3.0);
	    case 22: return 0.47308734787878004 * (xx - yy) * (7.0 * zz - 1.0);
	    case 23: return -1.7701307697799304 * xz * (xx - 3.0 * yy);
	    case 24: return 0.6258357354491761 * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy));
	}

	return 0.0;
}


// The real cuda forward_kernel
template <typename scalar_t>
__global__ void forward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> xyz,
    const uint32_t degree,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;  // col id
    const uint32_t n = blockIdx.y * blockDim.y + threadIdx.y;  // row id

    if (n < output.size(0) && c < output.size(1)) {
        output[n][c] = forward_table(c, xyz[n][0], xyz[n][1], xyz[n][2]);
    }
}


/* CUDA instantiate func for sh_encode forward
   @param: xyz, torch float tensor of (B, 3)
   @param: degree, int num
   @return: output, torch float tensor of (B, degree**2)
*/
torch::Tensor sh_encode_forward_cuda(
    const torch::Tensor xyz, const uint32_t degree) {
    torch::Tensor output = torch::zeros({xyz.size(0), square(degree)}).to(xyz.dtype()).to(xyz.device());

    const uint32_t n_row = output.size(0);  // B
    const uint32_t n_col = output.size(1);  // degree ** 2
    const dim3 threads(32, 32);
    const uint32_t thread_per_dim = 32;
    const dim3 blocks(div_round_up(n_col, thread_per_dim), div_round_up(n_row, thread_per_dim));

    // instantiate the real executable kernel
    AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "sh_encode_forward_cuda",
    ([&] {
        forward_kernel<scalar_t><<<blocks, threads>>>(
            xyz.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            degree,
            output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
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

    if (index == 0){
        switch(c){
            case 0: return 0.0;
            case 1: return 0.0;
            case 2: return 0.0;
            case 3: return -0.4886025119029199;
            case 4: return 1.0925484305920792 * y;
            case 5: return 0.0;
            case 6: return 0.0;
            case 7: return -1.0925484305920792 * z;
            case 8: return 0.5462742152960396 * 2.0 * x;
            case 9: return -0.5900435899266435 * 6.0 * xy;
	        case 10: return 2.890611442640554 * yz;
	        case 11: return 0.0;
	        case 12: return 0.0;
	        case 13: return -0.4570457994644658 * (5.0 * zz - 1.0);
	        case 14: return 1.445305721320277 * 2.0 * xz;
	        case 15: return -0.5900435899266435 * 3.0 * (xx - yy);
	        case 16: return 2.5033429417967046 * y * (3.0* xx - yy);
	        case 17: return -1.7701307697799304 * 6.0 * xyz;
	        case 18: return 0.9461746957575601 * y * (7.0 * zz - 1.0);
	        case 19: return 0.0;
	        case 20: return 0.0;
	        case 21: return -0.6690465435572892 * z * (7.0 * zz - 3.0);
	        case 22: return 0.47308734787878004 * 2.0 * x * (7.0 * zz - 1.0);
	        case 23: return -1.7701307697799304 * 3.0 * z * (xx - yy);
	        case 24: return 0.6258357354491761 * 4.0 * x * (xx - 3.0 * yy);
        }
    } else if (index == 1){
        switch(c){
            case 0: return 0.0;
            case 1: return -0.4886025119029199;
            case 2: return 0.0;
            case 3: return 0.0;
            case 4: return 1.0925484305920792 * x;
            case 5: return -1.0925484305920792 * z;
            case 6: return 0.0;
            case 7: return 0.0;
            case 8: return -0.5462742152960396 * 2.0 * y;
            case 9: return -0.5900435899266435 * 3.0 * (xx - yy);
	        case 10: return 2.890611442640554 * xz;
	        case 11: return -0.4570457994644658 * (5.0 * zz - 1.0);
            case 12: return 0.0;
            case 13: return 0.0;
	        case 14: return -1.445305721320277 * 2.0 * yz;
	        case 15: return 0.5900435899266435 * 6.0 * xy;
	        case 16: return 2.5033429417967046 * x * (xx - 3.0 * yy);
	        case 17: return -1.7701307697799304 * 3.0 * z * (xx - yy);
	        case 18: return 0.9461746957575601 * x * (7.0 * zz - 1.0);
	        case 19: return -0.6690465435572892 * z * (7.0 * zz - 3.0);
	        case 20: return 0.0;
	        case 21: return 0.0;
	        case 22: return 0.47308734787878004 * 2.0 * y * (1.0 - 7.0 * zz);
	        case 23: return 1.7701307697799304 * 6.0 * xyz;
	        case 24: return 0.6258357354491761 * 4.0 * y * (yy - 3.0 * xx);
        }
    } else if (index == 2) {
        switch(c){
            case 0: return 0.0;
            case 1: return 0.0;
            case 2: return 0.4886025119029199;
            case 3: return 0.0;
            case 4: return 0.0;
            case 5: return -1.0925484305920792 * y;
            case 6: return 0.31539156525252005 * 6.0 * z;
            case 7: return -1.0925484305920792 * x;
            case 8: return 0.0;
            case 9: return 0.0;
	        case 10: return 2.890611442640554 * xy;
	        case 11: return -0.4570457994644658 * 10.0 * yz;
	        case 12: return 0.3731763325901154 * (15.0 * zz - 3.0);
	        case 13: return -0.4570457994644658 * 10.0 * xz;
	        case 14: return 1.445305721320277 * (xx - yy);
	        case 15: return 0.0;
	        case 16: return 0.0;
	        case 17: return -1.7701307697799304 * y * (3.0 * xx - yy);
	        case 18: return 0.9461746957575601 * 14.0 * xyz;
	        case 19: return -0.6690465435572892 * 3.0 * y * (7.0 * zz - 1.0);
	        case 20: return 0.10578554691520431 * 20.0 * z * (7.0 * zz - 3.0);
	        case 21: return -0.6690465435572892 * 3.0 * x * (7.0 * zz - 1.0);
	        case 22: return 0.47308734787878004 * 14.0 * z * (xx - yy);
	        case 23: return -1.7701307697799304 * x * (xx - 3.0 * yy);
	        case 24: return 0.0;
        }
    }

    return 0.0;
}


// The real cuda backward_kernel
template <typename scalar_t>
__global__ void backward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> grad_out,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> xyz,
    const uint32_t degree,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> grad_xyz) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;  // col id
    const uint32_t n = blockIdx.y * blockDim.y + threadIdx.y;  // row id

    scalar_t x = xyz[n][0];
    scalar_t y = xyz[n][1];
    scalar_t z = xyz[n][2];

    if (n < grad_out.size(0) && c < grad_out.size(1)) {
        // Not sure when atomicAdd harms the performance
        atomicAdd(&grad_xyz[n][0], grad_out[n][c] * backward_table(c, 0, x, y, z));
        atomicAdd(&grad_xyz[n][1], grad_out[n][c] * backward_table(c, 1, x, y, z));
        atomicAdd(&grad_xyz[n][2], grad_out[n][c] * backward_table(c, 2, x, y, z));
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
    torch::Tensor grad_xyz = torch::zeros_like(xyz);  // (B, 3)

    const uint32_t n_row = grad_out.size(0);  // B
    const uint32_t n_col = grad_out.size(1);  // degree ** 2
    const dim3 threads(32, 32);  // 2d-block
    const uint32_t thread_per_dim = 32;
    const dim3 blocks(div_round_up(n_col, thread_per_dim), div_round_up(n_row, thread_per_dim));  // 2d-grid

    // instantiate the real executable kernel
    AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "sh_encode_backward_cuda",
    ([&] {
        backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            xyz.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            degree,
            grad_xyz.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return grad_xyz;
}
