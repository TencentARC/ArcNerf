# projection
Function for cam projection from 3d points into image pixels
- xyz_world -> xyz_cam -> pixel -> xyz_cam -> xyz_world
- xyz_world -> xyz_cam: w2c
- xyz_cam -> pixel: intrinsic
- pixel -> xyz_cam: intrinsic
- xyz_cam -> xyz_world: c2w

# transformation
Provide functions for geometrical transformation, including cam pose, vec, points
- normalize: norm vec
- rotation: rotate a matrix by R

# poses
Functions for create/modify cam poses
- invert pose: c2w <-> w2c transfer
