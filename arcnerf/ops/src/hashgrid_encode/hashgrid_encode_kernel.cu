// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// multi-res grid vertices hash embeddings

#include <torch/extension.h>

#include "helper.h"
#include "utils.h"


// hash function turning grid pts index to hash value, % by hashmap_size
template <uint32_t D>
__device__ uint32_t fast_hash(const uint32_t grid_pts_index[D], const uint32_t hashmap_size) {
    constexpr uint32_t primes[7] = {1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737};

    uint64_t hash = 0;

    # pragma unroll
    for (uint32_t i=0; i<D; i++) {
        hash ^= (uint64_t)grid_pts_index[i] * (uint64_t)primes[i];  // in case overflow
    }

    return (uint32_t)(hash % hashmap_size);
}


// The real cuda forward_kernel
template <typename scalar_t, uint32_t F, uint32_t D>
__global__ void forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz,  // (B, D)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> embeddings,  // (n_total_embed, F)
    const uint32_t L,  // n_levels
    const uint32_t* __restrict__ offsets,  // L+1
    const uint32_t* __restrict__ resolutions,  // L
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> min_xyz,  // D
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> max_xyz,  // D
    const bool cal_grad,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output,  // (B, L, F)
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,  // (B, L, 1<<D)
    torch::PackedTensorAccessor64<int64_t, 3, torch::RestrictPtrTraits> hash_idx,  // (B, L, 1<<D)
    torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> valid,  // (B,)
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dw_dxyz) {  // (B, L, 1<<D, D)

    const uint32_t level = blockIdx.x * blockDim.x + threadIdx.x;  // level=0,1,2,...,L-1
    const uint32_t n = blockIdx.y * blockDim.y + threadIdx.y;  // row id

    if (n < xyz.size(0) && level < L) {
        const uint32_t cur_res = (uint32_t)resolutions[level];
        const uint32_t hashmap_size = (uint32_t)(offsets[level+1] - offsets[level]);
        const uint32_t offset = (uint32_t)offsets[level];

        // grid size
        scalar_t voxel_size[D];
        # pragma unroll
        for (uint32_t i=0; i<D; i++) {
            voxel_size[i] = ((max_xyz[i] - min_xyz[i]) / (scalar_t)(cur_res));
        }

        // voxel index
        int32_t voxel_idx[D];
        # pragma unroll
        for (uint32_t i=0; i<D; i++) {
            voxel_idx[i] = (int32_t)((xyz[n][i] - min_xyz[i]) / voxel_size[i]);
        }

        // valid for this pts
        bool in_box = true;
        # pragma unroll
        for (uint32_t i=0; i<D; i++) {
            if (voxel_idx[i] < 0 || voxel_idx[i] >= cur_res)
                in_box = false;
        }

        if (in_box == false)
            return;  // it will not contribute to the embeddings

        if (cal_grad) {
            valid[n] = true;
        }

        // interpolate from every grid_pts in the voxel
        scalar_t w_xyz[D];  // left bottom grid_pts weights
        uint32_t grid_pts_idx[D];   // left bottom grid_pts index
        // grad for dw_x_dx
        scalar_t grad[D];

        if (cal_grad) {
            // grad for dw_x_dx
            # pragma unroll
            for (uint32_t d=0; d<D; d++) {
                grad[d] = 0.0;
            }
        }

        # pragma unroll
        for (uint32_t i=0; i<D; i++) {
            grid_pts_idx[i] = voxel_idx[i];
            scalar_t min_grid_pts = min_xyz[i] + (scalar_t)voxel_idx[i] * voxel_size[i];
            w_xyz[i] = (xyz[n][i] - min_grid_pts) / voxel_size[i];
            if (cal_grad) {
                if (w_xyz[i] >= 0.0 && w_xyz[i] <= 1.0){
                    grad[i] = 1.0 / voxel_size[i];
                }
            }
           w_xyz[i] = max(min(w_xyz[i], 1.0), 0.0);
        }

        // every grid_pts, find it's weight and feature
        # pragma unroll
        for (uint32_t i=0; i<(1<<D); i++) {
            scalar_t w = 1.0;
            uint32_t grid_pts_idx_local[D];

            // local grad dw_xyz_dw_x
            scalar_t grad_local[D];

            if (cal_grad) {
                // local grad dw_xyz_dw_x
                # pragma unroll
                for (uint32_t d=0; d<D; d++) {
                    grad_local[d] = 1.0;
                }
            }

            // for each dim, multi up the weights for this grid_pts
            # pragma unroll
            for (uint32_t d=0; d<D; d++) {  // each axis
                if (i & (1 << d)) {  // match the same axis
                    w *= w_xyz[d];
                    grid_pts_idx_local[d] = grid_pts_idx[d] + 1;

                    if (cal_grad) {
                        // grad from grid_pts to each dim
                        # pragma unroll
                        for (uint32_t k=0; k<D; k++) {
                            if (k != d) {
                                grad_local[k] *= w_xyz[d];
                            }
                        }
                    }
                } else {
                    w *= (1.0 - w_xyz[d]);
                    grid_pts_idx_local[d] = grid_pts_idx[d];

                    if (cal_grad) {
                        // grad from grid_pts to each dim
                        # pragma unroll
                        for (uint32_t k=0; k<D; k++) {
                            if (k == d) {
                                grad_local[k] *= -1;
                            } else {
                                grad_local[k] *= (1.0 - w_xyz[d]);
                            }
                        }
                    }
                }
            }

            if (cal_grad) {
                // update grad on each vertices to each dim
                # pragma unroll
                for (uint32_t d=0; d<D; d++) {
                    dw_dxyz[n][level][i][d] = grad[d] * grad_local[d];
                }
            }

            // hash_idx in hash bin
            uint32_t hash = fast_hash<D>(grid_pts_idx_local, hashmap_size) + offset;

            if (cal_grad) {
                hash_idx[n][level][i] = (int64_t)hash;
                weights[n][level][i] = w;
            }

            // for each feature, copy the weighted feature from grid pts
            # pragma unroll
            for (uint32_t f=0; f<F; f++) {
                atomicAdd(&output[n][level][f], embeddings[hash][f] * w);
            }
        }
    }

    return;
}


// The forward wrapper
template <typename scalar_t> void forward_kernel_wrapper(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz,  // (B, D)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> embeddings,  // (n_total_embed, F)
    const uint32_t L,  // n_levels
    const uint32_t F,  // n_feat_per_entry
    const uint32_t D,  // input dim
    const uint32_t* __restrict__ offsets,  // L+1
    const uint32_t* __restrict__ resolutions,  // L
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> min_xyz,  // D
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> max_xyz,  // D
    const bool cal_grad,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output,  // (B, L, F)
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,  // (B, L, 1<<D)
    torch::PackedTensorAccessor64<int64_t, 3, torch::RestrictPtrTraits> hash_idx,  // (B, L, 1<<D)
    torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> valid,  // (B,)
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dw_dxyz) {  // (B, L, 1<<D, D)

    const uint32_t batch_size = xyz.size(0);  // B
    const uint32_t threads_per_row = 512;
    const dim3 threads(1, threads_per_row);
    const dim3 blocks(L, div_round_up(batch_size, threads_per_row));

    // run the kernel
    switch (F) {
        case 1: {
            switch (D) {
                case 2: forward_kernel<scalar_t, 1, 2><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, cal_grad, output,
                            weights, hash_idx, valid, dw_dxyz
                        ); break;
                case 3: forward_kernel<scalar_t, 1, 3><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, cal_grad, output,
                            weights, hash_idx, valid, dw_dxyz
                        ); break;
                case 4: forward_kernel<scalar_t, 1, 4><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, cal_grad, output,
                            weights, hash_idx, valid, dw_dxyz
                        ); break;
                default: throw std::runtime_error{"Input dim must be 2, 3, 4."};
            }; break;
        } case 2: {
            switch (D) {
                case 2: forward_kernel<scalar_t, 2, 2><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, cal_grad, output,
                            weights, hash_idx, valid, dw_dxyz
                        ); break;
                case 3: forward_kernel<scalar_t, 2, 3><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, cal_grad, output,
                            weights, hash_idx, valid, dw_dxyz
                        ); break;
                case 4: forward_kernel<scalar_t, 2, 4><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, cal_grad, output,
                            weights, hash_idx, valid, dw_dxyz
                        ); break;
                default: throw std::runtime_error{"Input dim must be 2, 3, 4."};
            }; break;
        } case 4: {
            switch (D) {
                case 2: forward_kernel<scalar_t, 4, 2><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, cal_grad, output,
                            weights, hash_idx, valid, dw_dxyz
                        ); break;
                case 3: forward_kernel<scalar_t, 4, 3><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, cal_grad, output,
                            weights, hash_idx, valid, dw_dxyz
                        ); break;
                case 4: forward_kernel<scalar_t, 4, 4><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, cal_grad, output,
                            weights, hash_idx, valid, dw_dxyz
                        ); break;
                default: throw std::runtime_error{"Input dim must be 2, 3, 4."};
            }; break;
        } case 8: {
            switch (D) {
                case 2: forward_kernel<scalar_t, 8, 2><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, cal_grad, output,
                            weights, hash_idx, valid, dw_dxyz
                        ); break;
                case 3: forward_kernel<scalar_t, 8, 3><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, cal_grad, output,
                            weights, hash_idx, valid, dw_dxyz
                        ); break;
                case 4: forward_kernel<scalar_t, 8, 4><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, cal_grad, output,
                            weights, hash_idx, valid, dw_dxyz
                        ); break;
                default: throw std::runtime_error{"Input dim must be 2, 3, 4."};
            }; break;
        } default: throw std::runtime_error{"Feature per entry must be 1, 2, 4, 8."};
    }

    return;
}


/* CUDA instantiate func for hashgrid_encode forward
   @param: xyz, torch float tensor of (B, D)
   @param: embeddings, torch float tensor of (n_total_embed, F)
   @param: L, num of levels of embedding(L), by default 16
   @param: F, num of feat for each entry in hashmap(F), by default 2
   @param: offsets, torch float tensor of (L+1, ), offset of each level, len is L+1
   @param: resolutions, torch float tensor of (L, ), resolution at each level, len is L
   @param: min_xyz, torch float tensor of (D, ), the min_xyz position of the grid
   @param: max_xyz, torch float tensor of (D, ), the max_xyz position of the grid
   @param: cal_grad, bool value decide whether to cal grad
   @param: weights, torch float tensor of (B, L, 1<<D), the contributed weights in each level on each grid_pts
   @param: hash_idx, torch long tensor of (B, L, 1<<D), the hash index of pts in each level on each grid_pts
   @param: valid, torch bool tensor of (B,), whether the pts is in grid
   @param: dw_dxyz, torch float tensor of (B, L, 1<<D, D), save the grad from w_xyz to xyz
   @return: output, torch float tensor of (B, L, F)
*/
torch::Tensor hashgrid_encode_forward_cuda(
    const torch::Tensor xyz,
    const torch::Tensor embeddings,
    const uint32_t L,
    const uint32_t F,
    const std::vector<uint32_t> offsets,
    const std::vector<uint32_t> resolutions,
    const torch::Tensor min_xyz,
    const torch::Tensor max_xyz,
    const bool cal_grad,
    torch::Tensor weights,
    torch::Tensor hash_idx,
    torch::Tensor valid,
    torch::Tensor dw_dxyz) {

    const uint32_t B = xyz.size(0);  // B
    const uint32_t D = xyz.size(1);  // D

    // vector to gpu
    uint32_t* offsets_gpu = vec_to_gpu(offsets);
    uint32_t* resolutions_gpu = vec_to_gpu(resolutions);

    // Init the output tensor
    torch::Tensor output = torch::zeros({B, L, F}, xyz.dtype()).to(xyz.device());  // (B, L, F)

    // instantiate the real executable kernel
    AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "hashgrid_encode_forward_cuda",
    ([&] {
        forward_kernel_wrapper<scalar_t>(
            xyz.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            embeddings.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            L, F, D,
            offsets_gpu,
            resolutions_gpu,
            min_xyz.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            max_xyz.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            cal_grad,
            output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            hash_idx.packed_accessor64<int64_t, 3, torch::RestrictPtrTraits>(),
            valid.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
            dw_dxyz.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));

    return output;
}


// The real cuda backward_kernel
template <typename scalar_t, uint32_t F, uint32_t D>
__global__ void backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_out,  // (B, L, F)
    const uint32_t L,  // n_levels
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz,  // (B, D)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> embeddings,  // (n_total_embed, F)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,  // (B, L, 1<<D)
    const torch::PackedTensorAccessor64<int64_t, 3, torch::RestrictPtrTraits> hash_idx,  // (B, L, 1<<D)
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> valid,  // (B,)
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dw_dxyz,  // (B, L, 1<<D, D)
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_xyz,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_embeddings) {

    const uint32_t level = blockIdx.x * blockDim.x + threadIdx.x;  // level=0,1,2,...,L-1
    const uint32_t n = blockIdx.y * blockDim.y + threadIdx.y;  // row id

    if (n < xyz.size(0) && level < L) {
        if (valid[n] == true) {  // only update for the pts in volume

            # pragma unroll
            for (uint32_t f=0; f<F; f++) {

                # pragma unroll
                for (uint32_t i=0; i<(1<<D); i++) {
                    int64_t hash = hash_idx[n][level][i];  // hash index on certain grid_pts

                    // update grad on embedding
                    atomicAdd(&grad_embeddings[hash][f], grad_out[n][level][f] * weights[n][level][i]);

                    // update grad on xyz
                    # pragma unroll
                    for (uint32_t d=0; d<D; d++){
                        scalar_t accum_grad = grad_out[n][level][f] * embeddings[hash][f] * dw_dxyz[n][level][i][d];
                        atomicAdd(&grad_xyz[n][d], accum_grad);
                    }
                }
            }
        }
    }

    return;
}


// The backward wrapper
template <typename scalar_t> void backward_kernel_wrapper(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_out,  // (B, L, F)
    const uint32_t L,  // n_levels
    const uint32_t F,  // n_feat_per_entry
    const uint32_t D,  // input dim
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz,  // (B, D)
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> embeddings,  // (n_total_embed, F)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,  // (B, L, 1<<D)
    const torch::PackedTensorAccessor64<int64_t, 3, torch::RestrictPtrTraits> hash_idx,  // (B, L, 1<<D)
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> valid,  // (B,)
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dw_dxyz,  // (B, L, 1<<D, D)
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_xyz,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_embeddings) {

    const uint32_t batch_size = xyz.size(0);  // B
    const uint32_t threads_per_row = 512;
    const dim3 threads(1, threads_per_row);
    const dim3 blocks(L, div_round_up(batch_size, threads_per_row));

    // run the kernel
    switch (F) {
        case 1: {
            switch (D) {
                case 2: backward_kernel<scalar_t, 1, 2><<<blocks, threads>>>(
                            grad_out, L, xyz, embeddings, weights, hash_idx, valid, dw_dxyz, grad_xyz, grad_embeddings
                        ); break;
                case 3: backward_kernel<scalar_t, 1, 3><<<blocks, threads>>>(
                            grad_out, L, xyz, embeddings, weights, hash_idx, valid, dw_dxyz, grad_xyz, grad_embeddings
                        ); break;
                case 4: backward_kernel<scalar_t, 1, 4><<<blocks, threads>>>(
                            grad_out, L, xyz, embeddings, weights, hash_idx, valid, dw_dxyz, grad_xyz, grad_embeddings
                        ); break;
                default: throw std::runtime_error{"Input dim must be 2, 3, 4."};
            }; break;
        } case 2: {
            switch (D) {
                case 2: backward_kernel<scalar_t, 2, 2><<<blocks, threads>>>(
                            grad_out, L, xyz, embeddings, weights, hash_idx, valid, dw_dxyz, grad_xyz, grad_embeddings
                        ); break;
                case 3: backward_kernel<scalar_t, 2, 3><<<blocks, threads>>>(
                            grad_out, L, xyz, embeddings, weights, hash_idx, valid, dw_dxyz, grad_xyz, grad_embeddings
                        ); break;
                case 4: backward_kernel<scalar_t, 2, 4><<<blocks, threads>>>(
                            grad_out, L, xyz, embeddings, weights, hash_idx, valid, dw_dxyz, grad_xyz, grad_embeddings
                        ); break;
                default: throw std::runtime_error{"Input dim must be 2, 3, 4."};
            }; break;
        } case 4: {
            switch (D) {
                case 2: backward_kernel<scalar_t, 4, 2><<<blocks, threads>>>(
                            grad_out, L, xyz, embeddings, weights, hash_idx, valid, dw_dxyz, grad_xyz, grad_embeddings
                        ); break;
                case 3: backward_kernel<scalar_t, 4, 3><<<blocks, threads>>>(
                            grad_out, L, xyz, embeddings, weights, hash_idx, valid, dw_dxyz, grad_xyz, grad_embeddings
                        ); break;
                case 4: backward_kernel<scalar_t, 4, 4><<<blocks, threads>>>(
                            grad_out, L, xyz, embeddings, weights, hash_idx, valid, dw_dxyz, grad_xyz, grad_embeddings
                        ); break;
                default: throw std::runtime_error{"Input dim must be 2, 3, 4."};
            }; break;
        } case 8: {
            switch (D) {
                case 2: backward_kernel<scalar_t, 8, 2><<<blocks, threads>>>(
                            grad_out, L, xyz, embeddings, weights, hash_idx, valid, dw_dxyz, grad_xyz, grad_embeddings
                        ); break;
                case 3: backward_kernel<scalar_t, 8, 3><<<blocks, threads>>>(
                            grad_out, L, xyz, embeddings, weights, hash_idx, valid, dw_dxyz, grad_xyz, grad_embeddings
                        ); break;
                case 4: backward_kernel<scalar_t, 8, 4><<<blocks, threads>>>(
                            grad_out, L, xyz, embeddings, weights, hash_idx, valid, dw_dxyz, grad_xyz, grad_embeddings
                        ); break;
                default: throw std::runtime_error{"Input dim must be 2, 3, 4."};
            }; break;
        } default: throw std::runtime_error{"Feature per entry must be 1, 2, 4, 8."};
    }
}


/* CUDA instantiate func for hashgrid_encode backward
   @param: grad_out, torch float tensor of (B, L, F), final grad
   @param: xyz, torch float tensor of (B, D)
   @param: embeddings, torch float tensor of (n_total_embed, F)
   @param: weights, torch float tensor of (B, L, 1<<D), the contributed weights in each level on each grid_pts
   @param: hash_idx, torch long tensor of (B, L, 1<<D), the hash index of pts in each level on each grid_pts
   @param: valid, torch bool tensor of (B,), whether the pts is in grid
   @param: dw_dxyz, torch float tensor of (B, L, 1<<D, D), save the grad from w_xyz to xyz
   @return: list of output, first is grad_xyz (B, D), second is grad_embeddings (n_total_embed, F)
*/
std::vector<torch::Tensor> hashgrid_encode_backward_cuda(
    const torch::Tensor grad_out,
    const torch::Tensor xyz,
    const torch::Tensor embeddings,
    const torch::Tensor weights,
    const torch::Tensor hash_idx,
    const torch::Tensor valid,
    const torch::Tensor dw_dxyz) {

    const uint32_t B = xyz.size(0);  // B
    const uint32_t D = xyz.size(1);  // D
    const uint32_t L = grad_out.size(1);  // L
    const uint32_t F = grad_out.size(2);  // F

    // Init the output grad tensor
    torch::Tensor grad_xyz = torch::zeros_like(xyz).to(grad_out.device());  // (B, D)
    torch::Tensor grad_embeddings = torch::zeros_like(embeddings).to(grad_out.device());  // (n_total_embed, F)

    // instantiate the real executable kernel
    AT_DISPATCH_FLOATING_TYPES(grad_xyz.scalar_type(), "hashgrid_encode_backward_cuda",
    ([&] {
        backward_kernel_wrapper<scalar_t>(
            grad_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            L, F, D,
            xyz.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            embeddings.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            hash_idx.packed_accessor64<int64_t, 3, torch::RestrictPtrTraits>(),
            valid.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
            dw_dxyz.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            grad_xyz.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_embeddings.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    return {grad_xyz, grad_embeddings};
}
