#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>


// CUDA function for simple calculation on any type
template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}


/* CUDA instantiate func for hashgrid_encode forward
   @param: xyz, torch float tensor of (B, 3)
   @param: embeddings, torch float tensor of (n_total_embed, F)
   @param: grad_xyz
   @param: grad_embeddings
   @param: L, num of levels of embedding(L), by default 16
   @param: F, num of feat for each entry in hashmap(F), by default 2
   @param: offsets, a list of offset of each level, len is L+1
   @param: resolutions, a list of resolution at each level, len is L
   @param: min_xyz, a list of 3, the min_xyz position of the grid
   @param: max_xyz, a list of 3, the max_xyz position of the grid
   @return: output, torch float tensor of (B, L*F)
*/
torch::Tensor hashgrid_encode_forward_cuda(
    const torch::Tensor xyz,
    const torch::Tensor embeddings,
    torch::Tensor grad_xyz,
    torch::Tensor grad_embeddings,
    const uint32_t L,
    const uint32_t F,
    const std::vector<uint32_t> offsets,
    const std::vector<uint32_t> resolutions,
    const std::vector<float> min_xyz,
    const std::vector<float> max_xyz) {
    // Init the output tensor
    torch::Tensor output = torch::zeros({xyz.size(0), L * F}).to(xyz.dtype()).to(xyz.device());

    return output;
}


/* CUDA instantiate func for hashgrid_encode backward
   @param: grad_out, torch float tensor of (B, L*F), final grad
   @param: grad_xyz
   @param: grad_embeddings
   @return: list of output, first is grad_xyz (B, 3), second is grad_embeddings (n_total_embed, F)
*/
std::vector<torch::Tensor> hashgrid_encode_backward_cuda(
    const torch::Tensor grad, torch::Tensor grad_xyz, torch::Tensor grad_embeddings) {

    return {grad_xyz, grad_embeddings};
}
