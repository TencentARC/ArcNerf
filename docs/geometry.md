# projection
Function for cam projection from 3d points into image pixels
- xyz_world -> xyz_cam -> pixel -> xyz_cam -> xyz_world
- xyz_world -> xyz_cam: w2c(distortion is required)
- xyz_cam -> pixel: intrinsic
- pixel -> xyz_cam: intrinsic
- xyz_cam -> xyz_world: c2w

# transformation
Provide functions for geometrical transformation, including cam pose, vec, points
- normalize: norm vec
- rotation: rotate a matrix by R
- axis/rot representation interchange

# poses
Functions for create/modify cam poses
- invert pose: c2w <-> w2c transfer
- look_at/view_matrix: generate c2w
- average/center poses: get avg pose and recenter all poses

# sphere
Many function about sphere is provided, including:
- uv-xyz transform
- get sphere line/surface
- get spiral line

It provides help for camera path creation.

Any point on a unit sphere with (0,0,0) origin can be represented by (u, v),
where u in (0, 2pi), v in (0, pi)
- x = cos(u) * sin(v)
- y = cos(v)
- z = sin(u) * sin(v)

# ray
Functions for ray point.
- get_ray_point_by_zvals: get the real point on ray using rays_o/rays_d/zvals.

# triangle
Functions for triangle calculation.
- get tri normal
- get tri circumcircle
