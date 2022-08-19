# -*- coding: utf-8 -*-
# see: https://pytorch.org/tutorials/advanced/cpp_extension.html for details

import os
import os.path as osp
from setuptools import setup

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# compile on all arch
os.environ['TORCH_CUDA_ARCH_LIST'] = ''

include_dirs = [osp.dirname(osp.abspath(__file__)) + '/include']

setup(
    name='arcnerf_custom_ops',
    version='1.0',
    author='leoyluo',
    author_email='lawy623@gmail.com',
    description='custom cuda ops in arcnerf',
    long_description='custom cuda ops in arcnerf for view synthesis and 3d reconstruction',
    ext_modules=[
        CUDAExtension(  # spherical harmonics embedding
            name='_sh_encode',
            sources=['./src/sh_encode/sh_encode.cpp', './src/sh_encode/sh_encode_kernel.cu'],
            include_dirs=include_dirs,
        ),
        CUDAExtension(  # mul-res hashgrid embedding
            name='_hashgrid_encode',
            sources=['./src/hashgrid_encode/hashgrid_encode.cpp', './src/hashgrid_encode/hashgrid_encode_kernel.cu'],
            include_dirs=include_dirs
        )
    ],

    cmdclass={
        'build_ext': BuildExtension
    }
)
