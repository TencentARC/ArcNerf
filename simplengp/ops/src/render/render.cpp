// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// render related func


#include <torch/torch.h>

#include "utils.h"


// -------------------------------------------------- ------------------------------------ //


void calc_rgb_backward_cuda(
    const torch::Tensor sigma,
    const torch::Tensor radiance,
    const torch::Tensor rays_numsteps,
    const torch::Tensor dt,
    const torch::Tensor rgb_output,
    const torch::Tensor grad_rgb,
    torch::Tensor dloss_dsigma,
    torch::Tensor dloss_dradiance,
    const float early_stop);


void calc_rgb_backward(
    const torch::Tensor sigma,
    const torch::Tensor radiance,
    const torch::Tensor rays_numsteps,
    const torch::Tensor dt,
    const torch::Tensor rgb_output,
    const torch::Tensor grad_rgb,
    torch::Tensor dloss_dsigma,
    torch::Tensor dloss_dradiance,
    const float early_stop) {

    // checking
    CHECK_INPUT(sigma)
    CHECK_IS_FLOATING(sigma)
    CHECK_INPUT(radiance)
    CHECK_IS_FLOATING(radiance)
    CHECK_INPUT(rays_numsteps)
    CHECK_IS_INT(rays_numsteps)
    CHECK_INPUT(dt)
    CHECK_IS_FLOATING(dt)
    CHECK_INPUT(rgb_output)
    CHECK_IS_FLOATING(rgb_output)
    CHECK_INPUT(grad_rgb)
    CHECK_IS_FLOATING(grad_rgb)
    CHECK_INPUT(dloss_dsigma)
    CHECK_IS_FLOATING(dloss_dsigma)
    CHECK_INPUT(dloss_dradiance)
    CHECK_IS_FLOATING(dloss_dradiance)

    return calc_rgb_backward_cuda(
        sigma, radiance, rays_numsteps, dt, rgb_output,
        grad_rgb, dloss_dsigma, dloss_dradiance, early_stop
    );
}


// -------------------------------------------------- ------------------------------------ //


void calc_rgb_forward_cuda(
    const torch::Tensor sigma,
    const torch::Tensor radiance,
    const torch::Tensor rays_numsteps,
    const torch::Tensor dt,
    const torch::Tensor training_background_color,
    torch::Tensor rgb_output,
    torch::Tensor alpha_output,
    const float early_stop);

void calc_rgb_forward(
    const torch::Tensor sigma,
    const torch::Tensor radiance,
    const torch::Tensor rays_numsteps,
    const torch::Tensor dt,
    const torch::Tensor training_background_color,
    torch::Tensor rgb_output,
    torch::Tensor alpha_output,
    const float early_stop) {

    // checking
    CHECK_INPUT(sigma)
    CHECK_IS_FLOATING(sigma)
    CHECK_INPUT(radiance)
    CHECK_IS_FLOATING(radiance)
    CHECK_INPUT(rays_numsteps)
    CHECK_IS_INT(rays_numsteps)
    CHECK_INPUT(dt)
    CHECK_IS_FLOATING(dt)
    CHECK_INPUT(training_background_color)
    CHECK_IS_FLOATING(training_background_color)
    CHECK_INPUT(rgb_output)
    CHECK_IS_FLOATING(rgb_output)
    CHECK_INPUT(alpha_output)
    CHECK_IS_FLOATING(alpha_output)

    return calc_rgb_forward_cuda(sigma, radiance, rays_numsteps, dt,
            training_background_color, rgb_output, alpha_output, early_stop
    );
}


// -------------------------------------------------- ------------------------------------ //


void calc_rgb_inference_cuda(
    const torch::Tensor sigma,
    const torch::Tensor radiance,
    const torch::Tensor rays_numsteps,
    const torch::Tensor dt,
    const torch::Tensor bg_color,
    torch::Tensor rgb_output,
    torch::Tensor alpha_output,
    const float early_stop);

void calc_rgb_inference(
    const torch::Tensor sigma,
    const torch::Tensor radiance,
    const torch::Tensor rays_numsteps,
    const torch::Tensor dt,
    const torch::Tensor bg_color,
    torch::Tensor rgb_output,
    torch::Tensor alpha_output,
    const float early_stop) {

    // checking
    CHECK_INPUT(sigma)
    CHECK_IS_FLOATING(sigma)
    CHECK_INPUT(radiance)
    CHECK_IS_FLOATING(radiance)
    CHECK_INPUT(rays_numsteps)
    CHECK_IS_INT(rays_numsteps)
    CHECK_INPUT(dt)
    CHECK_IS_FLOATING(dt)
    CHECK_INPUT(bg_color)
    CHECK_IS_FLOATING(bg_color)
    CHECK_INPUT(rgb_output)
    CHECK_IS_FLOATING(rgb_output)
    CHECK_INPUT(alpha_output)
    CHECK_IS_FLOATING(alpha_output)

    return calc_rgb_inference_cuda(
        sigma, radiance, rays_numsteps, dt, bg_color, rgb_output, alpha_output, early_stop
    );
}

// -------------------------------------------------- ------------------------------------ //


void fill_input_forward_cuda(
    const torch::Tensor sigma,
    const torch::Tensor radiance,
    const torch::Tensor dt,
    const torch::Tensor rays_numsteps,
    torch::Tensor _sigma,
    torch::Tensor _radiance,
    torch::Tensor _dt);

void fill_input_forward(
    const torch::Tensor sigma,
    const torch::Tensor radiance,
    const torch::Tensor dt,
    const torch::Tensor rays_numsteps,
    torch::Tensor _sigma,
    torch::Tensor _radiance,
    torch::Tensor _dt) {

    // checking
    CHECK_INPUT(sigma)
    CHECK_IS_FLOATING(sigma)
    CHECK_INPUT(radiance)
    CHECK_IS_FLOATING(radiance)
    CHECK_INPUT(dt)
    CHECK_IS_FLOATING(dt)
    CHECK_INPUT(rays_numsteps)
    CHECK_IS_INT(rays_numsteps)
    CHECK_INPUT(_sigma)
    CHECK_IS_FLOATING(_sigma)
    CHECK_INPUT(_radiance)
    CHECK_IS_FLOATING(_radiance)
    CHECK_INPUT(_dt)
    CHECK_IS_FLOATING(_dt)

    return fill_input_forward_cuda(sigma, radiance, dt, rays_numsteps, _sigma, _radiance, _dt);
}


void fill_input_backward_cuda(
    torch::Tensor grad_sigma,
    torch::Tensor grad_radiance,
    torch::Tensor grad_dt,
    const torch::Tensor rays_numsteps,
    const torch::Tensor _grad_sigma,
    const torch::Tensor _grad_radiance,
    const torch::Tensor _grad_dt);

void fill_input_backward(
    torch::Tensor grad_sigma,
    torch::Tensor grad_radiance,
    torch::Tensor grad_dt,
    const torch::Tensor rays_numsteps,
    const torch::Tensor _grad_sigma,
    const torch::Tensor _grad_radiance,
    const torch::Tensor _grad_dt) {

    // checking
    CHECK_INPUT(grad_sigma)
    CHECK_IS_FLOATING(grad_sigma)
    CHECK_INPUT(grad_radiance)
    CHECK_IS_FLOATING(grad_radiance)
    CHECK_INPUT(grad_dt)
    CHECK_IS_FLOATING(grad_dt)
    CHECK_INPUT(rays_numsteps)
    CHECK_IS_INT(rays_numsteps)
    CHECK_INPUT(_grad_sigma)
    CHECK_IS_FLOATING(_grad_sigma)
    CHECK_INPUT(_grad_radiance)
    CHECK_IS_FLOATING(_grad_radiance)
    CHECK_INPUT(_grad_dt)
    CHECK_IS_FLOATING(_grad_dt)

    return fill_input_backward_cuda(
        grad_sigma, grad_radiance, grad_dt, rays_numsteps, _grad_sigma, _grad_radiance, _grad_dt
    );
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calc_rgb_inference", &calc_rgb_inference, "calc_rgb_inference (CUDA)");
    m.def("calc_rgb_forward", &calc_rgb_forward, "calc_rgb_forward (CUDA)");
    m.def("calc_rgb_backward", &calc_rgb_backward, "calc_rgb_backward (CUDA)");
    m.def("fill_input_forward", &fill_input_forward, "fill_input_forward (CUDA)");
    m.def("fill_input_backward", &fill_input_backward, "fill_input_backward (CUDA)");
}
