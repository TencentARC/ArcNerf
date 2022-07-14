# -*- coding: utf-8 -*-
# see: https://pytorch.org/tutorials/advanced/cpp_extension.html for details

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

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
            sources=['./src/sh_encode/sh_encode.cpp', './src/sh_encode/sh_encode_kernel.cu']
        ),
        CUDAExtension(  # mul-res hashgrid embedding
            name='_hashgrid_encode',
            sources=['./src/hashgrid_encode/hashgrid_encode.cpp', './src/hashgrid_encode/hashgrid_encode_kernel.cu']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
