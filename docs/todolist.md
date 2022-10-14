We may develop more function on this pipeline based on priority and time:

--------------------------------------------------------------------
Data preparation:
- Mask estimation
- Volume generation
- Maybe more accurate camera estimation

--------------------------------------------------------------------
Traditional Methods:
- We could add more MVS methods and mesh extraction methods

--------------------------------------------------------------------
Model level:
- Octree bound for pruning and sampling
- Deformable mesh templates for better mesh optimization(marching/deftet, tetrahedra, demtet)
- fully optimzation of the mesh output (nvdiffrec)
- compression (tensorRF)
- 360/background modeling (mipnerf-360)
- dynamic modeling(DNeRF and extension)
- decomposition of other components (light, texture) -> nerfFactor/nerd/neuraltex
- Other representation (Neural Volume, MSI/MPI)
- Human based modeling

--------------------------------------------------------------------
Optimization:
- MipNerf match the benchmark
- progressive training with sampling on errorous rays?
- mix-precision training


--------------------------------------------------------------------
Other functionality:
- More accurate and complete mesh extraction methods.
- nvdiffras based rendering optimzation
- real time train/inference online demo, any view point visual(opengl, cuda, etc)
- online training gui(many have done)
