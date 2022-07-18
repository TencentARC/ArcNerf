#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>


// CUDA function for simple calculation on any type
template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

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
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> xyz,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> embeddings,
    const uint32_t L,  // n_levels
    const torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> offsets,  // L+1
    const torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> resolutions,  // L
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> min_xyz,  // D
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> max_xyz,  // D
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> output) {

    const uint32_t n = blockIdx.y * blockDim.y + threadIdx.y;  // row id
    const uint32_t level = blockIdx.x * blockDim.x + threadIdx.x;  // level=0,1,2,...,L-1

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
        uint32_t voxel_idx[D];
        # pragma unroll
        for (uint32_t i=0; i<D; i++) {
            voxel_idx[i] = (uint32_t)((xyz[n][i] - min_xyz[i]) / voxel_size[i]);
        }

        // valid for this pts
        bool valid = true;
        # pragma unroll
        for (uint32_t i=0; i<D; i++) {
            if (voxel_idx[i] < 0 || voxel_idx[i] >= cur_res)
                valid = false;
        }

        if (valid == false)
            return;  // it will not contribute to the embeddings

        // interpolate from every grid_pts in the voxel
        scalar_t w_xyz[D];  // left bottom grid_pts weights
        uint32_t grid_pts_idx[D];   // left bottom grid_pts index

        # pragma unroll
        for (uint32_t i=0; i<D; i++) {
            grid_pts_idx[i] = voxel_idx[i];
            scalar_t min_grid_pts = min_xyz[i] + (scalar_t)voxel_idx[i] * voxel_size[i];
            w_xyz[i] = (xyz[n][i] - min_grid_pts) / voxel_size[i];
            w_xyz[i] = max(min(w_xyz[i], 1.0), 0.0);
        }

        // every grid_pts, find it's weight and feature
        # pragma unroll
        for (uint32_t i=0; i<(1<<D); i++) {
            scalar_t w = 1.0;
            uint32_t grid_pts_idx_local[D];

            // for each dim, multi up the weights for this grid_pts
            # pragma unroll
            for (uint32_t d=0; d<D; d++) {  // each axis
                if (i & (1 << d)) {  // match the same axis
                    w *= w_xyz[d];
                    grid_pts_idx_local[d] = grid_pts_idx[d] + 1;
                } else {
                    w *= (1.0 - w_xyz[d]);
                    grid_pts_idx_local[d] = grid_pts_idx[d];
                }
            }

            // hash_idx in hash bin
            uint32_t hash_idx = fast_hash<D>(grid_pts_idx_local, hashmap_size) + offset;

            // for each feature, copy the weighted feature from grid pts
            # pragma unroll
            for (uint32_t f=0; f<F; f++) {
                atomicAdd(&output[level][n][f], embeddings[hash_idx][f] * w);
            }
        }
    }

    return;
}


// The forward wrapper
template <typename scalar_t> void forward_kernel_wrapper(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> xyz,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> embeddings,
    const uint32_t L,  // n_levels
    const uint32_t F,  // n_feat_per_entry
    const uint32_t D,  // input dim
    const torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> offsets,  // L+1
    const torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> resolutions,  // L
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> min_xyz,  // D
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> max_xyz,  // D
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> output) {

    const uint32_t batch_size = xyz.size(0);  // B
    const uint32_t threads_per_row = 512;
    const dim3 threads(1, threads_per_row);
    const dim3 blocks(L, div_round_up(batch_size, threads_per_row));

    switch (F) {
        case 1: {
            switch (D) {
                case 2: forward_kernel<scalar_t, 1, 2><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, output
                        ); break;
                case 3: forward_kernel<scalar_t, 1, 3><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, output
                        ); break;
                case 4: forward_kernel<scalar_t, 1, 4><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, output
                        ); break;
                default: throw std::runtime_error{"Input dim must be 2,3,4."};
            }; break;
        } case 2: {
            switch (D) {
                case 2: forward_kernel<scalar_t, 2, 2><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, output
                        ); break;
                case 3: forward_kernel<scalar_t, 2, 3><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, output
                        ); break;
                case 4: forward_kernel<scalar_t, 2, 4><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, output
                        ); break;
                default: throw std::runtime_error{"Input dim must be 2,3,4."};
            }; break;
        } case 4: {
            switch (D) {
                case 2: forward_kernel<scalar_t, 4, 2><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, output
                        ); break;
                case 3: forward_kernel<scalar_t, 4, 3><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, output
                        ); break;
                case 4: forward_kernel<scalar_t, 4, 4><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, output
                        ); break;
                default: throw std::runtime_error{"Input dim must be 2,3,4."};
            }; break;
        } case 8: {
            switch (D) {
                case 2: forward_kernel<scalar_t, 8, 2><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, output
                        ); break;
                case 3: forward_kernel<scalar_t, 8, 3><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, output
                        ); break;
                case 4: forward_kernel<scalar_t, 8, 4><<<blocks, threads>>>(
                            xyz, embeddings, L, offsets, resolutions, min_xyz, max_xyz, output
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
   @param: grad_xyz
   @param: grad_embeddings
   @param: L, num of levels of embedding(L), by default 16
   @param: F, num of feat for each entry in hashmap(F), by default 2
   @param: offsets, torch float tensor of (L+1, ), offset of each level, len is L+1
   @param: resolutions, torch float tensor of (L, ), resolution at each level, len is L
   @param: min_xyz, torch float tensor of (D, ), the min_xyz position of the grid
   @param: max_xyz, torch float tensor of (D, ), the max_xyz position of the grid
   @return: output, torch float tensor of (B, L*F)
*/
torch::Tensor hashgrid_encode_forward_cuda(
    const torch::Tensor xyz,
    const torch::Tensor embeddings,
    torch::Tensor grad_xyz,
    torch::Tensor grad_embeddings,
    const uint32_t L,
    const uint32_t F,
    const torch::Tensor offsets,
    const torch::Tensor resolutions,
    const torch::Tensor min_xyz,
    const torch::Tensor max_xyz) {
    // Init the output tensor
    torch::Tensor output = torch::zeros({L, xyz.size(0), F}).to(xyz.dtype()).to(xyz.device());  // (L, B, F)

    const uint32_t D = xyz.size(1);  // D

    // instantiate the real executable kernel
    AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "hashgrid_encode_forward_cuda",
    ([&] {
        forward_kernel_wrapper<scalar_t>(
            xyz.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            embeddings.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            L, F, D,
            offsets.packed_accessor<int, 1, torch::RestrictPtrTraits, size_t>(),
            resolutions.packed_accessor<int, 1, torch::RestrictPtrTraits, size_t>(),
            min_xyz.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            max_xyz.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()
        );
    }));

    // reshape
    output = output.permute({1, 0, 2});  // (B, L, F)
    output = output.contiguous();
    output = output.view({output.size(0), -1});  // (B, L*F)

    return output;
}


/* CUDA instantiate func for hashgrid_encode backward
   @param: grad_out, torch float tensor of (B, L*F), final grad
   @param: grad_xyz
   @param: grad_embeddings
   @return: list of output, first is grad_xyz (B, D), second is grad_embeddings (n_total_embed, F)
*/
std::vector<torch::Tensor> hashgrid_encode_backward_cuda(
    const torch::Tensor grad, torch::Tensor grad_xyz, torch::Tensor grad_embeddings) {

    return {grad_xyz, grad_embeddings};
}
