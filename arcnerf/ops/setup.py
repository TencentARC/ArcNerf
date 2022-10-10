# -*- coding: utf-8 -*-
# see: https://pytorch.org/tutorials/advanced/cpp_extension.html for details

import os
from setuptools import setup

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# compile on all arch
os.environ['TORCH_CUDA_ARCH_LIST'] = ''

include_dirs = ['./include', './include/eigen']

setup(
    name='arcnerf_custom_ops',
    version='1.0',
    author='leoyluo',
    author_email='lawy623@gmail.com',
    description='custom cuda ops in arcnerf',
    long_description='custom cuda ops in arcnerf for view synthesis and 3d reconstruction',
    include_dirs=include_dirs,
    ext_modules=[
        CUDAExtension(  # volume related func
            name='_volume_func',
            sources=['./src/volume_func/volume_func.cpp', './src/volume_func/volume_func_kernel.cu'],
        ),
    ],

    cmdclass={
        'build_ext': BuildExtension
    }
)
