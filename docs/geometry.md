------------------------------------------------------------------------
# projection
Function for cam projection from 3d points into image pixels
- `xyz_world` -> `xyz_cam` -> `pixel` -> `xyz_cam` -> `xyz_world`
- `xyz_world` -> `xyz_cam`: w2c
- `xyz_cam` -> `pixel`: intrinsic
- `pixel` -> `xyz_cam`: intrinsic
- `xyz_cam` -> `xyz_world`: c2w
- Distortion is allowed in `world->pixel`

------------------------------------------------------------------------
# transformation
Provide functions for geometrical transformation, including cam pose, vec, points
- normalize: norm vec
- rotation: rotate a matrix by R
- axis/rot representation interchange

------------------------------------------------------------------------
# poses
Functions for create/modify cam poses
- invert pose: `c2w` <-> `w2c` transfer
- look_at/view_matrix: generate `c2w`
- average/center poses: get avg pose and recenter all poses
- get cam on sphere by different mode

------------------------------------------------------------------------
# sphere
Many function about sphere is provided, including:
- uv-xyz transform
- get sphere line/surface
- get spiral line
- get swing line

It provides help for camera path creation.

Any point on a unit sphere with `(0,0,0)` origin can be represented by `(u, v)`,
where `u` in `(0, 2pi)`, `v` in `(0, pi)`
- x = cos(u) * sin(v)
- y = cos(v)
- z = sin(u) * sin(v)

------------------------------------------------------------------------
# ray
Functions for ray point. ray is `(rays_o, rays_d)`, rays_d is always assumed to be normalized.
- get_ray_point_by_zvals: get the real point on ray using `rays_o/rays_d/zvals`.
- closest_point_on_ray: find the closest point on ray to an existing point. zvals can not be negative.
- closest_point_to_two_rays: two rays and their closest pts pair with distance. All case applied(parallel, zvals<0)
- closest_distance_of_two_rays: distance of two ray. But need to assume rays are pointing inward.
- closest_point_to_rays: a point close to all rays. Good for cam view centralization.

------------------------------------------------------------------------
# triangle
Functions for triangle calculation.
- get_tri_normal
- get_tri_circumcircle

------------------------------------------------------------------------
# mesh
Function for mesh extraction, color extraction, etc.
- extract_mesh: simple marching cubes on cpu
- save_meshes: save mesh to `.ply` files
- get_verts_by_faces: rearrange verts from `(V, 3)` to `(F, 3, 3)` by faces.
- get_normals: get the vert/face normals
- get_face_centers: get the triangle centers
- simplify_mesh: simplify mesh
- render_mesh: it will call open3d/pytorch3d backend to render the mesh by cam positions.

------------------------------------------------------------------------
# volume
Definition and function of a volume. For all the point you can get it in grid or fatten(by default)
- n_grid: num of volume/line seg on each side
- corner: 8 corner pts
- grid_pts: all the grid points, in total `(n_grid+1)^3`
- voxel_size: each voxel size, useful for marching cubes.
- volume_pts: all the volume center points, in total `(n_grid)^3`
  - This can be sent to the network and get the volume density
- bound_lines: outside bounding lines, `12` lines with `(2, 3)` start-end pts.
- dense_lines: inner+outside bounding lines, `3*(n+1)^3` lines with `(2, 3)` start-end pts.
- bound_faces: outside bounding faces, `6 faces`, tensor in `(6, 4, 3)` shape
- dense_faces: inner+outside bounding faces, tensor in `((n_grid+1)n_grid^2*3, 4, 3)` shape
- convert_flatten_index_to_xyz_index/convert_xyz_index_to_flatten_index: index conversion
## ray/pts in volume
For ray in pts in volume, we provide a lot of function like
- check_pts_in_grid_boundary: check pts in voxel
- get_voxel_idx_from_xyz: get voxel idx from pts position
- get_grid_pts_idx_by_voxel_idx/get_grid_pts_by_voxel_idx: get grid pts index and position by voxel idx
- cal_weights_to_grid_pts / interpolate: interpolate pts by grid_pts using trilinear interpolation

------------------------------------------------------------------------
# point cloud
Function of point cloud with pts and color.
- save_point_cloud: export pc file as .ply file
