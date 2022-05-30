# ArcNerf


# TODO:
- cuda实现的geometry函数和数据结构
- tetrahedra相关
- nvdiffras vs pytorch3d (The renderer is not fully tested yet. Inconsistency appears in rendering)

- 光照拆解
- 材质拆解

- real time inference online demo, any view point visual(opengl, cuda, etc)
- 在线训练可视化， 前后端

- 参考框架
  - nerf(nerfpl) - DONE
  - nerf++ - DONE
  - neus/volsdf (sdf/occpancy) - neurcon - DONE

  - nsvf/plenoxel/plenoctree + instangp/hash_nerf
  - neuraltex
  - marching/deftet, demtet
  - nerfFactor/nerd
  - mip-nerf
  - dynamic
  - neural volume
  - nex (mpi, msi, etc)


# Ref
- https://github.com/kwea123/nerf_pl
- https://github.com/yenchenlin/nerf-pytorch
- https://github.com/ventusff/neurecon
- https://github.com/Kai-46/nerfplusplus
- https://github.com/Totoro97/NeuS
- https://github.com/lioryariv/volsdf
- https://github.com/yashbhalgat/HashNeRF-pytorch

# dataset
- nerf/LLFF: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
- DTU/BlenderMVS: https://www.dropbox.com/sh/oum8dyo19jqdkwu/AAAxpIifYjjotz_fIRBj1Fyla
