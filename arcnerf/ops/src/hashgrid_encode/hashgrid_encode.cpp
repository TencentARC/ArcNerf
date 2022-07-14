#include <torch/extension.h>
#include <torch/torch.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || \
                                         x.scalar_type() == at::ScalarType::Half || \
                                         x.scalar_type() == at::ScalarType::Double, \
                                         #x " must be a floating tensor")


// define the real cuda function to be called by c++ wrapper.
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
    const std::vector<float> max_xyz);


/* c++ wrapper of hashgrid_encode forward func
   py: hashgrid_encode_forward(xyz, embeddings, grad_xyz, grad_embeddings, n_levels, n_feat_per_entry,
                               offsets, resolutions, min_xyz, max_xyz)
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
torch::Tensor hashgrid_encode_forward(
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
    //checking
    CHECK_INPUT(xyz)
    CHECK_IS_FLOATING(xyz)
    CHECK_INPUT(embeddings)
    CHECK_IS_FLOATING(embeddings)
    CHECK_INPUT(grad_xyz)
    CHECK_IS_FLOATING(grad_xyz)
    CHECK_INPUT(grad_embeddings)
    CHECK_IS_FLOATING(grad_embeddings)

    if (xyz.size(1) != 3){
        throw std::runtime_error{"Input xyz tensor must be (B, 3)."};
    }
    if (grad_xyz.size(1) != 3){
        throw std::runtime_error{"Input grad xyz tensor must be (B, 3)."};
    }

    if (offsets.size() != L + 1){
        throw std::runtime_error{"Offset length must be L+1."};
    }

    int n_total_embed = offsets[L];
    if (embeddings.size(0) != n_total_embed || embeddings.size(1) != F){
        throw std::runtime_error{"embeddings tensor must be (n_total_embed, F)."};
    }
    if (grad_embeddings.size(0) != n_total_embed || grad_embeddings.size(1) != F){
        throw std::runtime_error{"grad embeddings tensor must be (n_total_embed, F)."};
    }

    if (resolutions.size() != L){
        throw std::runtime_error{"Resolutions length must be L."};
    }

    if (min_xyz.size() != 3 || max_xyz.size() != 3){
        throw std::runtime_error{"xyz boundary length must be 3."};
    }

    // call actual cuda function
    return hashgrid_encode_forward_cuda(xyz, embeddings, grad_xyz, grad_embeddings, L, F,
                                        offsets, resolutions, min_xyz, max_xyz);
}

// define the real cuda function to be called by c++ wrapper.
std::vector<torch::Tensor> hashgrid_encode_backward_cuda(
    const torch::Tensor grad, torch::Tensor grad_xyz, torch::Tensor grad_embeddings);


/* c++ wrapper of hashgrid_encode backward func
   py: hashgrid_encode_bacwardward(grad, grad_xyz, grad_embeddings)
   @param: grad_out, torch float tensor of (B, L*F), final grad
   @param: grad_xyz
   @param: grad_embeddings
   @return: list of output, first is grad_xyz (B, 3), second is grad_embeddings (n_total_embed, F)
*/
std::vector<torch::Tensor> hashgrid_encode_backward(
    const torch::Tensor grad_out,
    torch::Tensor grad_xyz,
    torch::Tensor grad_embeddings) {
    //checking
    CHECK_INPUT(grad_out)
    CHECK_IS_FLOATING(grad_out)
    CHECK_INPUT(grad_xyz)
    CHECK_IS_FLOATING(grad_xyz)
    CHECK_INPUT(grad_embeddings)
    CHECK_IS_FLOATING(grad_embeddings)

    // call actual cuda function
    return hashgrid_encode_backward_cuda(grad_out, grad_xyz, grad_embeddings);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hashgrid_encode_forward", &hashgrid_encode_forward, "hashgrid encode forward (CUDA)");
    m.def("hashgrid_encode_backward", &hashgrid_encode_backward, "hashgrid encode backward (CUDA)");
}
