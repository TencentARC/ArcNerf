# -*- coding: utf-8 -*-

import os
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# compile on all arch
os.environ['TORCH_CUDA_ARCH_LIST'] = ''

# setup function
setup(
    name='simplengp_custom_ops',
    version='1.0',
    author='leoyluo',
    author_email='lawy623@gmail.com',
    description='custom cuda ops in instant-ngp',
    long_description='custom cuda ops in simplengp',
    include_dirs=[
        './include', './include/eigen'
    ],
    # compile each module
    ext_modules=[
        CUDAExtension(
            name='_dense_grid',
            sources=['./src/dense_grid/dense_grid.cpp', './src/dense_grid/dense_grid_kernel.cu'],
            extra_compile_args={'nvcc': ['--extended-lambda', '--expt-relaxed-constexpr']},
        ),
        CUDAExtension(
            name='_sampler',
            sources=['./src/sampler/sampler.cpp', './src/sampler/sampler_kernel.cu'],
        ),
        CUDAExtension(
            name='_render',
            sources=['./src/render/render.cpp', './src/render/render_kernel.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension})
