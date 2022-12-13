// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// render related func

#include <torch/extension.h>

#include "common.h"


// -------------------------------------------------- ------------------------------------ //

template <typename TYPE>
__global__ void calc_rgb_forward_cuda_kernel(
    const uint32_t n_rays,                      //batch total rays number
    const TYPE *sigma,                          //sigma output
    const TYPE *radiance,                       //radiance output
    uint32_t *__restrict__ numsteps_in,         //rays offset and base counter
    const float *__restrict__ dt,
    Array3f *rgb_output,                        //rays rgb output
    float *alpha_output,                        //rays alpha output
    const Array3f *bg_color_ptr,                //background color
    const float early_stop
    )
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_rays) { return; }

    Array3f background_color=bg_color_ptr[i];
    uint32_t numsteps = numsteps_in[i * 2 + 0];
    uint32_t base = numsteps_in[i * 2 + 1];
    if (numsteps == 0)
    {
        rgb_output[i] = background_color;
        alpha_output[i] = 0;
        return;
    }
    sigma += base;
    radiance += base * 3;
    dt += base;

    float T = 1.f;

    Array3f rgb_ray = Array3f::Zero();

    uint32_t compacted_numsteps = 0;
    for (; compacted_numsteps < numsteps; ++compacted_numsteps)
    {
        const Array3f rgb(radiance[0], radiance[1], radiance[2]);
        const float alpha = 1.f - __expf(-sigma[0] * dt[0]);
        const float weight = alpha * T;
        rgb_ray += weight * rgb;

        T *= (1.f - alpha);
        if (early_stop > 0.0 && T < early_stop) { break; }  // do early_stop

        // move ptr
        sigma += 1;
        radiance += 3;
    }

    // append background
    rgb_ray += T * background_color;

    rgb_output[i] = rgb_ray;
    alpha_output[i] = 1-T;
}

void calc_rgb_forward_cuda(
    const torch::Tensor sigma,
    const torch::Tensor radiance,
    const torch::Tensor rays_numsteps,
    const torch::Tensor dt,
    const torch::Tensor training_background_color,
    torch::Tensor rgb_output,
    torch::Tensor alpha_output,
    const float early_stop) {

    cudaStream_t stream = 0;

    float* sigma_p = (float*)sigma.data_ptr();
    float* radiance_p = (float*)radiance.data_ptr();
    float* dt_p = (float*)dt.data_ptr();
    uint32_t* rays_numsteps_p = (uint32_t*)rays_numsteps.data_ptr();
    Array3f* training_background_color_p = (Array3f*)training_background_color.data_ptr();

    // output
    Array3f* rgb_output_p = (Array3f*)rgb_output.data_ptr();
    float* alpha_output_p = (float*)alpha_output.data_ptr();

    const uint32_t n_rays = rays_numsteps.sizes()[0];

    linear_kernel(calc_rgb_forward_cuda_kernel<float>, 0, stream,
        n_rays, sigma_p, radiance_p, rays_numsteps_p, dt_p, rgb_output_p, alpha_output_p,
        training_background_color_p, early_stop);

    cudaDeviceSynchronize();
}

// -------------------------------------------------- ------------------------------------ //

template <typename TYPE>
__global__ void calc_rgb_backward_cuda_kernel(
    const uint32_t n_rays,                      //batch total rays number
    TYPE *__restrict__ dloss_dsigma,            //dloss_dsigma, grad on sigma
    TYPE *__restrict__ dloss_dradiance,         //dloss_dradiance, grad on radiance
    const TYPE *sigma,                          //sigma output
    const TYPE *radiance,                       //radiance output
    uint32_t *__restrict__ numsteps_in,         //rays offset and base counter after compact
    const float *__restrict__ dt,
    Array3f *__restrict__ loss_grad,            //dloss_dRGBoutput
    Array3f *__restrict__ rgb_ray,               //RGB from forward calculation
    const float early_stop
    )
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_rays) { return; }

    uint32_t numsteps = numsteps_in[i * 2 + 0];
    uint32_t base = numsteps_in[i * 2 + 1];

    sigma += base;
    radiance += base * 3;
    dt += base;
    dloss_dsigma += base;
    dloss_dradiance += base * 3;

    loss_grad += i;
    rgb_ray += i;

    float T = 1.f;
    uint32_t compacted_numsteps = 0;
    Array3f rgb_ray2 = Array3f::Zero();
    for (; compacted_numsteps < numsteps; ++compacted_numsteps)
    {
        const Array3f rgb(radiance[0], radiance[1], radiance[2]);
        const float alpha = 1.f - __expf(-sigma[0] * dt[0]);
        const float weight = alpha * T;
        rgb_ray2 += weight * rgb;
        T *= (1.f - alpha);

        const Array3f suffix = *rgb_ray - rgb_ray2;
        const Array3f dloss_by_drgb = weight * (*loss_grad);

        // write loss rgb
        dloss_dradiance[0] = dloss_by_drgb.x();
        dloss_dradiance[1] = dloss_by_drgb.y();
        dloss_dradiance[2] = dloss_by_drgb.z();

        // write loss sigma
        float dloss_by_dsigma = dt[0] * (*loss_grad).matrix().dot((T * rgb - suffix).matrix());
        dloss_dsigma[0] = dloss_by_dsigma;

        if (early_stop > 0.0 && T < early_stop) { break; } // do early_stop

        sigma += 1;
        radiance += 3;
        dloss_dsigma += 1;
        dloss_dradiance += 3;
    }
}


void calc_rgb_backward_cuda(
    const torch::Tensor sigma,
    const torch::Tensor radiance,
    const torch::Tensor rays_numsteps,
    const torch::Tensor dt,
    const torch::Tensor rgb_output,
    const torch::Tensor grad_rgb,
    torch::Tensor dloss_dsigma,
    torch::Tensor dloss_dradiance,
    const float early_stop) {

    cudaStream_t stream = 0;

    // input
    float* sigma_p = (float*)sigma.data_ptr();
    float* radiance_p = (float*)radiance.data_ptr();
    float* dt_p = (float*)dt.data_ptr();
    uint32_t* rays_numsteps_p = (uint32_t*)rays_numsteps.data_ptr();
    Array3f* grad_rgb_p = (Array3f*)grad_rgb.data_ptr();
    Array3f* rgb_output_p = (Array3f*)rgb_output.data_ptr();

    // output
    float* dloss_dsigma_p = (float*)dloss_dsigma.data_ptr();
    float* dloss_dradiance_p = (float*)dloss_dradiance.data_ptr();

    const uint32_t n_rays = rays_numsteps.sizes()[0];

    linear_kernel(calc_rgb_backward_cuda_kernel<float>, 0, stream,
        n_rays, dloss_dsigma_p, dloss_dradiance_p,
        sigma_p, radiance_p, rays_numsteps_p, dt_p,
        grad_rgb_p, rgb_output_p, early_stop);

    cudaDeviceSynchronize();
}


// -------------------------------------------------- ------------------------------------ //


template <typename TYPE>
__global__ void calc_rgb_inference_cuda_kernel(
    const uint32_t n_rays,                       //batch total rays number
    Array3f *__restrict__ background_color,      //same background color
    const TYPE *sigma,                           //sigma output
    const TYPE *radiance,                        //radiance output
    uint32_t *__restrict__ numsteps_in,          //rays offset and base counter
    const float *__restrict__ dt,                // dt
    Array3f *__restrict__ rgb_output,            //rays rgb output
    float* __restrict__ alpha_output,
    const float early_stop
    )
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n_rays) { return; }

    uint32_t numsteps = numsteps_in[i * 2 + 0];
    uint32_t base = numsteps_in[i * 2 + 1];
    if (numsteps == 0)
    {
        rgb_output[i] = *background_color;
        alpha_output[i] = 0;
        return;
    }
    sigma += base;
    radiance += base * 3;
    dt += base;

    float T = 1.f;

    Array3f rgb_ray = Array3f::Zero();

    uint32_t compacted_numsteps = 0;
    for (; compacted_numsteps < numsteps; ++compacted_numsteps)
    {
        const Array3f rgb(radiance[0], radiance[1], radiance[2]);
        const float alpha = 1.f - __expf(-sigma[0] * dt[0]);
        const float weight = alpha * T;
        rgb_ray += weight * rgb;

        T *= (1.f - alpha);
        if (early_stop > 0.0 && T < early_stop) { break; }  // do early_stop

        // move ptr
        sigma += 1;
        radiance += 3;
    }

    rgb_ray += T * (*background_color);

    rgb_output[i] = rgb_ray;
    alpha_output[i] = 1-T;
}

void calc_rgb_inference_cuda(
    const torch::Tensor sigma,
    const torch::Tensor radiance,
    const torch::Tensor rays_numsteps,
    const torch::Tensor dt,
    const torch::Tensor bg_color,
    torch::Tensor rgb_output,
    torch::Tensor alpha_output,
    const float early_stop) {

    cudaStream_t stream = 0;
    // input
    float* sigma_p = (float*)sigma.data_ptr();
    float* radiance_p = (float*)radiance.data_ptr();
    float* dt_p = (float*)dt.data_ptr();
    uint32_t* rays_numsteps_p = (uint32_t*)rays_numsteps.data_ptr();
    Array3f* bg_color_p = (Array3f*)bg_color.data_ptr();

    // output
    Array3f* rgb_output_p = (Array3f*)rgb_output.data_ptr();
    float* alpha_output_p = (float*)alpha_output.data_ptr();

    const uint32_t n_rays = rays_numsteps.sizes()[0];

    linear_kernel(calc_rgb_inference_cuda_kernel<float>, 0, stream,
        n_rays, bg_color_p, sigma_p, radiance_p, rays_numsteps_p, dt_p,
        rgb_output_p, alpha_output_p, early_stop);

    cudaDeviceSynchronize();
}


// -------------------------------------------------- ------------------------------------ //

__global__ void fill_input_forward_cuda_kernel(
    const uint32_t n_rays,                      //batch total rays number
    const uint32_t n_pts,                      //n_pts each ray
    const float *sigma,                          //sigma output
    const float *radiance,                       //radiance output
    const float *dt,
    uint32_t *__restrict__ numsteps_in,         //rays offset and base counter
    float *_sigma,
    float *_radiance,
    float *_dt)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_rays) { return; }

    uint32_t numsteps = numsteps_in[i * 2 + 0];
    uint32_t base = numsteps_in[i * 2 + 1];
    if (numsteps == 0)
    {
        return;
    }
    sigma += base;
    radiance += base * 3;
    dt += base;

    _sigma += n_pts * i;
    _radiance += n_pts * i * 3;
    _dt += n_pts * i;

    for (uint32_t k=0; k < numsteps; ++k) {
        _sigma[k] = sigma[k];
        _radiance[k * 3] = radiance[k * 3];
        _radiance[k * 3 + 1] = radiance[k * 3 + 1];
        _radiance[k * 3 + 2] = radiance[k * 3 + 2];
        _dt[k] = dt[k];
    }
}

void fill_input_forward_cuda(
    const torch::Tensor sigma,
    const torch::Tensor radiance,
    const torch::Tensor dt,
    const torch::Tensor rays_numsteps,
    torch::Tensor _sigma,
    torch::Tensor _radiance,
    torch::Tensor _dt) {

    cudaStream_t stream = 0;

    // input
    float* sigma_p = (float*)sigma.data_ptr();
    float* radiance_p = (float*)radiance.data_ptr();
    float* dt_p = (float*)dt.data_ptr();
    uint32_t* rays_numsteps_p = (uint32_t*)rays_numsteps.data_ptr();

    // output
    float* _sigma_p = (float*)_sigma.data_ptr();
    float* _radiance_p = (float*)_radiance.data_ptr();
    float* _dt_p = (float*)_dt.data_ptr();

    const uint32_t n_rays = _sigma.sizes()[0];
    const uint32_t max_n_pts = _sigma.sizes()[1];

    linear_kernel(fill_input_forward_cuda_kernel, 0, stream,
        n_rays, max_n_pts, sigma_p, radiance_p, dt_p, rays_numsteps_p, _sigma_p, _radiance_p, _dt_p);

    cudaDeviceSynchronize();
}


// -------------------------------------------------- ------------------------------------ //

__global__ void fill_input_backward_cuda_kernel(
    const uint32_t n_rays,                      //batch total rays number
    const uint32_t n_pts,                      //n_pts each ray
    float *grad_sigma,
    float *grad_radiance,
    float *grad_dt,
    uint32_t *__restrict__ numsteps_in,         //rays offset and base counter
    const float *_grad_sigma,
    const float *_grad_radiance,
    const float *_grad_dt)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_rays) { return; }

    uint32_t numsteps = numsteps_in[i * 2 + 0];
    uint32_t base = numsteps_in[i * 2 + 1];
    if (numsteps == 0)
    {
        return;
    }
    grad_sigma += base;
    grad_radiance += base * 3;
    grad_dt += base;

    _grad_sigma += n_pts * i;
    _grad_radiance += n_pts * i * 3;
    _grad_dt += n_pts * i;

    for (uint32_t k=0; k < numsteps; ++k) {
        grad_sigma[k] = _grad_sigma[k];
        grad_radiance[k * 3] = _grad_radiance[k * 3];
        grad_radiance[k * 3 + 1] = _grad_radiance[k * 3 + 1];
        grad_radiance[k * 3 + 2] = _grad_radiance[k * 3 + 2];
        grad_dt[k] = _grad_dt[k];
    }
}

void fill_input_backward_cuda(
    torch::Tensor grad_sigma,
    torch::Tensor grad_radiance,
    torch::Tensor grad_dt,
    const torch::Tensor rays_numsteps,
    const torch::Tensor _grad_sigma,
    const torch::Tensor _grad_radiance,
    const torch::Tensor _grad_dt) {

    cudaStream_t stream = 0;

    // input
    float* _grad_sigma_p = (float*)_grad_sigma.data_ptr();
    float* _grad_radiance_p = (float*)_grad_radiance.data_ptr();
    float* _grad_dt_p = (float*)_grad_dt.data_ptr();
    uint32_t* rays_numsteps_p = (uint32_t*)rays_numsteps.data_ptr();

    // output
    float* grad_sigma_p = (float*)grad_sigma.data_ptr();
    float* grad_radiance_p = (float*)grad_radiance.data_ptr();
    float* grad_dt_p = (float*)grad_dt.data_ptr();

    const uint32_t n_rays = _grad_sigma.sizes()[0];
    const uint32_t max_n_pts = _grad_sigma.sizes()[1];

    linear_kernel(fill_input_backward_cuda_kernel, 0, stream,
        n_rays, max_n_pts, grad_sigma_p, grad_radiance_p, grad_dt_p,
         rays_numsteps_p, _grad_sigma_p, _grad_radiance_p, _grad_dt_p);

    cudaDeviceSynchronize();
}