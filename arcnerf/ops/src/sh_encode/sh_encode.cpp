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
torch::Tensor sh_encode_forward_cuda(torch::Tensor xyz, const uint32_t degree);


/* c++ wrapper of sh_encode forward func
   py: sh_encode_forward(xyz, degree)
   @param: xyz, torch float tensor of (B, 3)
   @param: degree, int num
   @return: output, torch float tensor of (B, degree**2)
*/
torch::Tensor sh_encode_forward(torch::Tensor xyz, const uint32_t degree) {
    //checking
    CHECK_INPUT(xyz)
    CHECK_IS_FLOATING(xyz)

    if (xyz.size(1) != 3){
        throw std::runtime_error{"Input tensor must be (B, 3)."};
    }

    if (degree <= 0 || degree > 5){
        throw std::runtime_error{"Only support degree in 1~5."};
    }

    // call actual cuda function
    return sh_encode_forward_cuda(xyz, degree);
}


// define the real cuda function to be called by c++ wrapper.
torch::Tensor sh_encode_backward_cuda(
    torch::Tensor grad_out, torch::Tensor xyz, const uint32_t degree);


/* c++ wrapper of sh_encode backward func
   py: sh_encode_backward(grad, xyz, degree)
   @param: grad_out, torch float tensor of (B, degree**2), final grad
   @param: xyz, torch float tensor of (B, 3)
   @param: degree, int num
   @return: grad_xyz, torch float tensor of (B, 3)
*/
torch::Tensor sh_encode_backward(
    torch::Tensor grad_out,
    torch::Tensor xyz,
    const uint32_t degree) {
    //checking
    CHECK_INPUT(xyz)
    CHECK_INPUT(grad_out)

    CHECK_IS_FLOATING(xyz)
    CHECK_IS_FLOATING(grad_out)

    if (xyz.size(1) != 3){
        throw std::runtime_error{"Input tensor must be (B, 3)."};
    }

    if (degree <= 0 || degree > 5){
        throw std::runtime_error{"Only support degree in 1~5."};
    }

    // call actual cuda function
    return sh_encode_backward_cuda(grad_out, xyz, degree);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sh_encode_forward", &sh_encode_forward, "sh encode forward (CUDA)");
    m.def("sh_encode_backward", &sh_encode_backward, "sh encode backward (CUDA)");
}
