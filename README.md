# ArcNerf

------------------------------------------------------------------------
# TODO:
- cuda实现的geometry函数和数据结构(Some are good)
- tetrahedra相关
- nvdiffras vs pytorch3d (The renderer is not fully tested yet. Inconsistency appears in rendering)

- mix-precision
- 光照拆解
- 材质拆解

- real time inference online demo, any view point visual(opengl, cuda, etc)
- 在线训练可视化， 前后端

- 参考框架
  - nerf(nerfpl) - DONE
  - nerf++ - DONE
  - neus/volsdf (sdf/occpancy) - neurcon - DONE
  - mip-nerf - Implemented. Benchmark not reach
  - instangp/hash_nerf
    - ff_mlp/gridencoder/shencoder/raychingmarching（torch版本， tinycudann版本， torchngp版本）
  - nsvf/plenoxel/plenoctree
  - tensorRF

  - mip-nerf-360
  - diver (voxel interval representation)
  - neuraltex
  - marching/deftet, demtet
  - nerfFactor/nerd
  - dynamic
  - neural volume
  - nex (mpi, msi, etc)

- thinking:
  - progressive training with sampling on errorous rays?

------------------------------------------------------------------------
# Installation
git clone --recursive xxx

## Colmap

------------------------------------------------------------------------
# Usage

------------------------------------------------------------------------
# What is special in this project?

------------------------------------------------------------------------
## Datasets and Benchmarks

### Self-Capture data and Colmap


------------------------------------------------------------------------
## Models


### full_model


### Base_3d_model

------------------------------------------------------------------------
## Geometry

------------------------------------------------------------------------
## Visualization

------------------------------------------------------------------------
## Code and Tests
Sufficient docs, detail shape of each tensor

------------------------------------------------------------------------
## Trainer



------------------------------------------------------------------------
# Citation
Please see [Citation](docs/citation.md) for citations.

If you find this project useful, please consider citing:
```
TODO:
```
