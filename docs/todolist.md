We may develop more function on this pipeline based on priority and time:

--------------------------------------------------------------------
Data preparation:
- Mask estimation
- Volume generation/3d bbox estimation
- Maybe more accurate camera estimation

--------------------------------------------------------------------
Traditional Methods:
- We could add more MVS methods and mesh extraction methods

--------------------------------------------------------------------
Model level:
- Octree bound for pruning and sampling（will it be better than density volume?)
- Deformable mesh templates for better mesh optimization(marching/deftet, tetrahedra, demtet)
- fully optimization of the mesh output (nvdiffrec)
- compression (tensorRF)
- 360/background modeling (mipnerf-360)
- dynamic modeling(DNeRF and extension)
- decomposition of other components (light, texture) -> nerfFactor/nerd/neuraltex
- Other representation (Neural Volume, MSI/MPI)
- Human based modeling

--------------------------------------------------------------------
Optimization:
- progressive training with sampling on errorous rays?
- mix-precision training


--------------------------------------------------------------------
Other functionality:
- More accurate and complete mesh extraction methods.
- nvdiffras based rendering optimization
- real time train/inference online demo, any view point visual(opengl, cuda, etc)
- better online training gui with data processing(end-to-end pipeline)
- 3d modification, text2obj generation(HOT TOPIC)
